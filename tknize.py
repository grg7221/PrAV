from datasets import load_dataset
from tiktoken import get_encoding
import numpy as np
import torch

CPU_cores = 6
tk = get_encoding('gpt2') # токенайзер gpt2, vocab_size ~50k 
ds = load_dataset("roneneldan/TinyStories", split="train") # берем исходный датасет

# Функция токенизации отдельного батча (примера) датасета
def tokenize(batch):
    return {
        "tokens": [tk.encode(x) for x in batch['text']]
    }

def main():
    # токенизируем весь датасет (с помощью map прогоняем каждый пример через функцию tokenize), исходный текст удаляем
    ds_tokenized = ds.map(tokenize, batched=True, num_proc=CPU_cores, remove_columns='text')
    ds_tokenized.save_to_disk('E:/dataset/tokenized')
    print("Токенизация завершена")
    print("Вид первого примера из ДС:", ds_tokenized['tokens'][0])
    # результат - датасет, в котором один столбец "tokens", хранящий токенизированный исходный текст каждого примера

    # Извлекаем данные из полученного столбца
    tokens = ds_tokenized['tokens']
    # Превращаем каждый набор токенов в массив NumPy (быстрее, чем Python-list, но все еще очень медленно)
    ds_tensoried = np.concatenate([np.array(x, dtype=np.int32) for x in tokens])
    print("Конкатенация токенов в один NP-массив завершена: ", ds_tensoried)

    # Создаем одномерный тензор из всех токенов датасета, хранящихся в NumPy-массиве
    x = torch.from_numpy(ds_tensoried).cuda()
    print("Массив NumPy преобразован в одномерный тензор torch:", x[:100])

    # сохраняем в виде torch-тензора для загрузки готового массива в VRAM при обучении
    torch.save(x, 'E:/dataset/tensoried/ds_tokens.pt')
    print('Тензор датасета сохранен.')

if __name__ == "__main__":
    main()