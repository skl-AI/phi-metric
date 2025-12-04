from google.colab import drive
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
import faiss
import numpy as np
import math

# 1. ПОДКЛЮЧЕНИЕ GOOGLE DRIVE
drive.mount('/content/drive')
BASE_OUTPUT_DIR = "/content/drive/MyDrive/LLM_IDIAS_Experiments" 

# Создаем базовую папку, если она не существует
os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

# --- 2. НАСТРОЙКА ПАРАМЕТРОВ ЭКСПЕРИМЕНТА ---

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
MAX_SEQ_LENGTH = 512 # На T4 можно безопасно увеличить до 512 или 1024
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05

# 🚨 ПАРАМЕТРЫ, КОТОРЫЕ НУЖНО МЕНЯТЬ ДЛЯ ЗАПУСКОВ: 🚨
# 1. Контроль: 0.0
# 2. Эксперимент 1: 0.1
# 3. Эксперимент 2: 0.5
PHI_LAMBDA = 0.25 # <--- !!! МЕНЯТЬ ДЛЯ КАЖДОГО ЗАПУСКА !!!

PHI_SCALE_FACTOR = 10000.0
FAISS_DIMENSION = 256 # Оставляем 256 для снижения нагрузки на ЦП (хотя T4 справляется лучше)

# ПУТЬ ДЛЯ СОХРАНЕНИЯ (ДОЛЖЕН МЕНЯТЬСЯ В ЗАВИСИМОСТИ ОТ PHI_LAMBDA)
output_folder_name = f"adapters_lambda_{str(PHI_LAMBDA).replace('.', '_')}"
FINAL_OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, output_folder_name)
os.makedirs(FINAL_OUTPUT_DIR, exist_ok=True)

print(f"Результаты будут сохранены в: {FINAL_OUTPUT_DIR}")

# --- 1. Настройка QLoRA и Модели (TinyLlama-1.1B) ---

# Конфигурация 4-битного квантования
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# Загрузка модели и токенайзера
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto"
)
model.config.use_cache = False
model = prepare_model_for_kbit_training(model)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

# Конфигурация QLoRA для Llama-архитектуры
lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
)
model = get_peft_model(model, lora_config)

# Загрузка и подготовка данных
data = load_dataset("Abirate/english_quotes")

tokenized_data = data.map(
    lambda samples: tokenizer(samples["quote"], max_length=MAX_SEQ_LENGTH, truncation=True, padding="max_length"), 
    batched=True
)

# --- 2. Custom Trainer ---
class PhiTrainer(Trainer):
    def __init__(self, *args, phi_lambda=0.1, phi_scale_factor=10000.0, faiss_dimension=256, **kwargs):
        super().__init__(*args, **kwargs)
        self.phi_lambda = phi_lambda
        self.phi_scale_factor = phi_scale_factor
        self.faiss_dimension = faiss_dimension
        self.faiss_index = self._create_faiss_index()
        self.k_neighbors = 5 

    def _create_faiss_index(self):
        embed_layer = self.model.base_model.model.model.embed_tokens
        embedding_weights = embed_layer.weight.data.float().cpu().numpy()
        d = embedding_weights.shape[1]
        
        if d > self.faiss_dimension:
            embedding_weights = embedding_weights[:, :self.faiss_dimension]
        
        index = faiss.IndexFlatL2(self.faiss_dimension)
        index.add(embedding_weights)
        print(f"FAISS Index создан. Размерность: {self.faiss_dimension} (исходная: {d})")
        return index

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        
        if inputs.get("labels") is None:
            inputs["labels"] = inputs["input_ids"]

        # 1. СТАНДАРТНЫЙ РАСЧЕТ ПОТЕРЬ (NLL)
        outputs = model(**inputs, output_hidden_states=True)
        loss_nll = outputs.loss
        
        # --- 2. РАСЧЕТ РЕГУЛЯРИЗАТОРА КОГЕРЕНТНОСТИ (L_phi) ---
        last_hidden_states = outputs.hidden_states[-1] 
        
        # Уменьшаем размерность для CPU/FAISS
        H = last_hidden_states[..., :self.faiss_dimension] 
        H = H.view(-1, self.faiss_dimension).detach().cpu() 
        
        active_tokens_mask = (inputs['labels'] != -100).view(-1).cpu().numpy()
        H_numpy = H.float().numpy()
        active_states = H_numpy[active_tokens_mask]
        
        if active_states.shape[0] == 0:
            return (loss_nll, outputs) if return_outputs else loss_nll

        # Расчет H_i (Неопределенность/Complexity Loss)
        D, I = self.faiss_index.search(active_states, self.k_neighbors) 
        H_i = torch.tensor(np.mean(D, axis=1)).to(last_hidden_states.device)
        
        # Расчет I_i (Прогностическая сила/Information Gain)
        I_i = 1.0 / (torch.log(H_i + 1e-6) + 1.0) 
        
        # --- 2.4. РАСЧЕТ ФИ-МЕТРИКИ И НОВОГО РЕГУЛЯРИЗАТОРА ОТКЛОНЕНИЯ ---
        
        # ФИ-МЕТРИКА (PHI_i) = I_i / H_i
        Phi_tokens = I_i / H_i 
        
        # НОВАЯ ПОТЕРЯ: Квадратичное отклонение от целевого значения 1: (Phi_i - 1)^2
        L_phi_deviation_tokens = (Phi_tokens - 1.0).pow(2)
        
        # Усредняем и масштабируем новую потерю
        L_phi_deviation = L_phi_deviation_tokens.mean() / self.phi_scale_factor 
        
        # -------------------------------------------------------------
        
        # 3. ИТОГОВАЯ ФУНКЦИЯ ПОТЕРЬ (L_total = L_NLL + lambda * L_phi_deviation)
        loss_total = loss_nll + self.phi_lambda * L_phi_deviation
        
        # 4. ЛОГИРОВАНИЕ 
        if self.state.global_step % self.args.logging_steps == 0:
             self.log({
                 'loss_nll': loss_nll.item(), 
                 'loss_phi_deviation': L_phi_deviation.item(),
                 'avg_phi_metric': Phi_tokens.mean().item() 
             })
        
        return (loss_total, outputs) if return_outputs else loss_total

# --- 3. Настройка аргументов обучения ---
training_args = TrainingArguments(
    output_dir=FINAL_OUTPUT_DIR, # <--- СКОРРЕКТИРОВАНО
    num_train_epochs=3,
    per_device_train_batch_size=4, # <--- УВЕЛИЧЕНО для T4
    gradient_accumulation_steps=2, # <--- УМЕНЬШЕНО для T4 (эффективный batch_size = 8)
    learning_rate=2e-4,
    fp16=True, 
    logging_steps=10
)

# --- 4. Инициализация и Запуск ---
train_dataset = tokenized_data["train"]

trainer = PhiTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    phi_lambda=PHI_LAMBDA,
    phi_scale_factor=PHI_SCALE_FACTOR,
    faiss_dimension=FAISS_DIMENSION 
)

print(f"Запуск обучения TinyLlama с phi_lambda = {PHI_LAMBDA}. Градиенты L_phi будут влиять на LoRA-веса.")

trainer.train()

# Сохранение обученных LoRA-адаптеров
model.save_pretrained(FINAL_OUTPUT_DIR) # <--- СКОРРЕКТИРОВАНО
tokenizer.save_pretrained(FINAL_OUTPUT_DIR) # <--- СКОРРЕКТИРОВАНО

print(f"Обучение завершено. LoRA-адаптеры сохранены в {FINAL_OUTPUT_DIR}.")