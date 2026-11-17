import torch

def run_tensors_homework():
    """Скрипт для выполнения Задания 1: Создание и манипуляции с тензорами."""
    print("=" * 60)
    print("ЗАДАНИЕ 1: СОЗДАНИЕ И МАНИПУЛЯЦИИ С ТЕНЗОРАМИ")
    print("=" * 60)

    # --- 1.1 Создание тензоров ---
    print("\n--- 1.1 Создание тензоров ---")
    t_rand = torch.rand(3, 4)                     # 3x4 от 0 до 1
    t_zeros = torch.zeros(2, 3, 4)                # 2x3x4 нули
    t_ones = torch.ones(5, 5)                     # 5x5 единицы
    t_seq = torch.arange(16).reshape(4, 4)        # 4x4 с числами от 0 до 15

    print(f"1. Тензор 3x4 (случайные numbers [0, 1)):\n{t_rand}\n")
    print(f"2. Тензор 2x3x4 (нули):\n{t_zeros}\n")
    print(f"3. Тензор 5x5 (единицы):\n{t_ones}\n")
    print(f"4. Тензор 4x4 (от 0 до 15):\n{t_seq}\n")

    # --- 1.2 Операции с тензорами ---
    print("\n--- 1.2 Операции с тензорами ---")
    A = torch.rand(3, 4)
    B = torch.rand(4, 3)

    # Валидация размерностей перед операциями
    assert A.shape == (3, 4) and B.shape == (4, 3), "Неверные размерности входных тензоров!"

    A_transposed = A.T                             # Транспонирование A (станет 4x3)
    matmul_result = torch.matmul(A, B)             # Матричное умножение (3x4) @ (4x3) -> (3x3)
    elementwise_mult = A * B.T                     # Поэлементное умножение (3x4) * (3x4) -> (3x4)
    sum_A = A.sum()                                # Сумма всех элементов A

    print(f"Тензор A (3x4):\n{A}")
    print(f"Транспонированный A (4x3):\n{A_transposed}")
    print(f"Матричное умножение A @ B (3x3):\n{matmul_result}")
    print(f"Поэлементное умножение A * B.T (3x4):\n{elementwise_mult}")
    print(f"Сумма всех элементов A: {sum_A.item():.4f}\n")

    # --- 1.3 Индексация и срезы ---
    print("\n--- 1.3 Индексация и срезы ---")
    T = torch.rand(5, 5, 5)

    first_row = T[0, 0, :]                         # Первая строка (первый слой, первая строка)
    last_col = T[:, :, -1]                          # Последний столбец
    center_2x2 = T[2, 1:3, 1:3]                     # Подматрица 2x2 из центра тензора
    even_indices = T[::2, ::2, ::2]                 # Все элементы с четными индексами (шаг 2)

    print(f"Форма исходного 3D-тензора: {T.shape}")
    print(f"Первая строка T[0, 0, :]:\n{first_row}")
    print(f"Форма последнего столбца T[:, :, -1]: {last_col.shape}")
    print(f"Центральная подматрица 2x2:\n{center_2x2}")
    print(f"Элементы с четными индексами (форма): {even_indices.shape}\n")

    # --- 1.4 Работа с формами ---
    print("\n--- 1.4 Работа с формами ---")
    tensor_24 = torch.arange(24)
    print(f"Исходный вектор (24 элемента): {tensor_24.shape}")

    target_shapes = [(2, 12), (3, 8), (4, 6), (2, 3, 4), (2, 2, 2, 3)]
    for shape in target_shapes:
        reshaped = tensor_24.reshape(shape)
        # Проверка сохранения количества элементов
        assert reshaped.numel() == 24, f"Ошибка при преобразовании в форму {shape}"
        print(f"Преобразовано в {shape}: итоговая форма {reshaped.shape}")

if __name__ == "__main__":
    run_tensors_homework()