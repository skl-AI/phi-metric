import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# 1. ОСНОВНЫЕ ПАРАМЕТРЫ (НЕДОСТАЮЩИЙ БЛОК)
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
BASE_OUTPUT_DIR = "/content/drive/MyDrive/LLM_IDIAS_Experiments" 

# Список папок с адаптерами для сравнения
EXPERIMENT_FOLDERS = [
    "adapters_lambda_0_25_seq_phi",   # Контроль
#    "adapters_lambda_0_1",   # Эксперимент 1
#   "adapters_lambda_0_25",  # <-- ДОБАВЛЕНО: Новый адаптер 
#    "adapters_lambda_0_5"   # Эксперимент 2
]

# Промты для тестирования (ИЗМЕНЕНЫ НА ФОРМАТ CHAT)
# TinyLlama использует шаблон: <|system|>\n<|user|>\n{prompt}\n<|assistant|>\n
# Мы подадим ей только часть <|user|>\n{prompt}\n<|assistant|>\n
TEST_PROMPTS = [
"<|user|>\nExplain the process of photosynthesis in three main steps, starting with light absorption.\n<|assistant|>\n",
"<|user|>\nProvide a historical timeline (4 events) of the development of the internet from ARPANET to the World Wide Web.\n<|assistant|>\n",
"<|user|>\nExplain the fundamental differences between John Dewey's concept of experience and Immanuel Kant's categorical imperative. Mention their core works.\n<|assistant|>\n",
"<|user|>\nList the main differences between the philosophical schools of Stoicism and Epicureanism regarding the pursuit of happiness.\n<|assistant|>\n",
"<|user|>\nDefine the concept of 'Eigenvalue' and provide a simple, real-world example of its application in data analysis.\n<|assistant|>\n",
"<|user|>\nProvide a structured framework (pros/cons analysis, goal alignment) for making a decision between high-risk investment and stable savings.\n<|assistant|>\n",
"<|user|>\nMy team member always agrees in meetings but misses deadlines. Give me three tactical steps to resolve this sensitive conflict.\n<|assistant|>\n",
"<|user|>\nExplain the 'Theory of Constraints' and its four main steps for improving a system.\n<|assistant|>\n",
"<|user|>\nDesign a five-stage maturity model for an AI-adoption strategy in a large enterprise, focusing on Governance and Data Infrastructure.\n<|assistant|>\n",
"<|user|>\nProvide a SWOT analysis (Strengths, Weaknesses, Opportunities, Threats) for an early-stage startup entering a competitive market.\n<|assistant|>\n",
"<|user|>\nIn a world where thoughts were indeed geometrical shapes, describe what a moment of profound insight would look look like.\n<|assistant|>\n",
"<|user|>\nDescribe the concept of emergence in complex systems using the analogy of a symphony orchestra or a flock of birds.\n<|assistant|>\n",
"<|user|>\nAssume the persona of a famous minimalist writer and describe the process of reducing ideas to their core essence.\n<|assistant|>\n",
"<|user|>\nExplain the meaning of the word 'saudade' (a deep emotional state) using only abstract metaphors, not direct definitions.\n<|assistant|>\n",
"<|user|>\nWhat is the philosophical difference between 'knowledge' and 'understanding', expressed as a visual paradox?\n<|assistant|>\n",

]

# Параметры генерации (МОДИФИЦИРОВАНЫ ДЛЯ УСПЕШНОЙ ГЕНЕРАЦИИ)
GENERATION_KWARGS = {
    "max_new_tokens": 150,    # Увеличиваем
    "do_sample": True,
    "temperature": 0.3,       # Снижаем
    "top_k": 50
}

# 2. ЗАГРУЗКА БАЗОВОЙ МОДЕЛИ И ТОКЕНАЙЗЕРА (МОДИФИЦИРОВАНО ДЛЯ CPU)
print("Загрузка базовой модели и токенайзера...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

# ВАЖНО: Мы не используем bnb_config для CPU, но нам нужно его определение
# чтобы избежать ошибки NameError в блоке 4 (Цикл сравнения)
class DummyBitsAndBytesConfig:
    pass
if 'BitsAndBytesConfig' not in globals():
    BitsAndBytesConfig = DummyBitsAndBytesConfig

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    # quantization_config=bnb_config, # ЗАКОММЕНТИРОВАНО
    # device_map="auto" 
)

# Перемещаем модель на CPU, так как GPU недоступен
base_model.to("cpu")

# 3. ФУНКЦИЯ ТЕСТИРОВАНИЯ (МОДИФИЦИРОВАНО ДЛЯ CPU)
def run_generation(model, tokenizer, prompt, kwargs):
    # Теперь тензоры отправляются на CPU
    inputs = tokenizer(prompt, return_tensors="pt").to("cpu") 
    
    # Генерация
    with torch.no_grad():
        output_ids = model.generate(**inputs, **kwargs)
        
    # Декодирование
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return output_text[len(prompt):].strip()


# 4. ЦИКЛ СРАВНЕНИЯ
print("\n--- НАЧАЛО СРАВНЕНИЯ ---")
for folder_name in EXPERIMENT_FOLDERS:
    adapter_path = os.path.join(BASE_OUTPUT_DIR, folder_name)
    
    # 🟢 ИЗМЕНЕННАЯ ЛОГИКА: Извлекаем значение лямбда для вывода
    if "seq_phi" in folder_name:
        # Новый формат: adapters_lambda_X_Y_seq_phi
        parts = folder_name.split('_')
        # Собираем лямбду и добавляем метку
        lambda_value = parts[2] + '.' + parts[3] + ' (SEQ PHI)'
    else:
        # Старый формат: adapters_lambda_X_Y
        lambda_value = folder_name.split('_')[-2] + '.' + folder_name.split('_')[-1]
    
    if not os.path.exists(adapter_path):
        print(f"⚠️ Папка не найдена: {folder_name}. Пропускаю.")
        continue
        
    print(f"\n################################################################")
    print(f"## ТЕСТИРОВАНИЕ: LAMBDA = {lambda_value}")
    print(f"################################################################")

    # Загрузка PEFT-адаптера
    model = PeftModel.from_pretrained(base_model, adapter_path)
    # Объединяем, чтобы избежать ошибок
    model = model.merge_and_unload() 
    model.eval()

    for i, prompt in enumerate(TEST_PROMPTS):
        print(f"\n[Промт {i+1}]: {prompt}")
        
        # Запуск генерации
        response = run_generation(model, tokenizer, prompt, GENERATION_KWARGS)
        
        # Вывод результата
        print(f"Response ({lambda_value}) >> {response}")

    # Очистка для следующей итерации
    del model
    # torch.cuda.empty_cache() # Не нужно, так как работаем на CPU
    # Перезагружаем базовую модель на CPU
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        # quantization_config=bnb_config, # Исключено
        # device_map="auto" # Исключено
    ).to("cpu")

print("\n--- СРАВНЕНИЕ ЗАВЕРШЕНО ---")