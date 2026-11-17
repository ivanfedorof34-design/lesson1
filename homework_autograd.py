import torch

def homework_autograd():
    """Скрипт для выполнения Задания 2: Автоматическое дифференцирование."""
    print("=" * 60)
    print("ЗАДАНИЕ 2: АВТОМАТИЧЕСКОЕ ДИФФЕРЕНЦИРОВАНИЕ")
    print("=" * 60)

    # --- 2.1 Простые вычисления с градиентами ---
    print("\n--- 2.1 Простые вычисления с градиентами ---")
    x = torch.tensor(2.0, requires_grad=True)
    y = torch.tensor(3.0, requires_grad=True)
    z = torch.tensor(1.5, requires_grad=True)

    # f(x, y, z) = x^2 + y^2 + z^2 + 2*x*y*z
    f = x**2 + y**2 + z**2 + 2 * x * y * z
    f.backward()

    # Аналитический расчет:
    # df/dx = 2x + 2yz  = 2(2) + 2(3)(1.5) = 4 + 9 = 13
    # df/dy = 2y + 2xz  = 2(3) + 2(2)(1.5) = 6 + 6 = 12
    # df/dz = 2z + 2xy  = 2(1.5) + 2(2)(3) = 3 + 12 = 15
    grad_x_ana = 2 * x.item() + 2 * y.item() * z.item()
    grad_y_ana = 2 * y.item() + 2 * x.item() * z.item()
    grad_z_ana = 2 * z.item() + 2 * x.item() * y.item()

    print(f"Значение функции f(x,y,z): {f.item()}")
    print(f"df/dx: PyTorch = {x.grad.item():.2f} | Аналитически = {grad_x_ana:.2f}")
    print(f"df/dy: PyTorch = {y.grad.item():.2f} | Аналитически = {grad_y_ana:.2f}")
    print(f"df/dz: PyTorch = {z.grad.item():.2f} | Аналитически = {grad_z_ana:.2f}")

    assert torch.isclose(x.grad, torch.tensor(grad_x_ana)), "Градиент x не совпал!"

    # --- 2.2 Градиент функции потерь (MSE) ---
    print("\n--- 2.2 Градиент функции потерь (MSE) ---")
    def compute_mse(w: torch.Tensor, b: torch.Tensor, x_data: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """Реализация функции MSE: (1/n) * sum((y_pred - y_true)^2), где y_pred = w * x + b."""
        y_pred = w * x_data + b
        return torch.mean((y_pred - y_true) ** 2)

    w = torch.tensor(0.5, requires_grad=True)
    b = torch.tensor(0.1, requires_grad=True)

    x_data = torch.tensor([1.0, 2.0, 3.0, 4.0])
    y_true = torch.tensor([2.0, 4.0, 6.0, 8.0])

    loss = compute_mse(w, b, x_data, y_true)
    loss.backward()

    print(f"Значение MSE Loss: {loss.item():.4f}")
    print(f"Градиент по весу w (dL/dw): {w.grad.item():.4f}")
    print(f"Градиент по смещению b (dL/db): {b.grad.item():.4f}")

    # --- 2.3 Цепное правило ---
    print("\n--- 2.3 Цепное правило ---")
    x_chain = torch.tensor(1.0, requires_grad=True)
    f_chain = torch.sin(x_chain**2 + 1)

    # Вычисление градиента через torch.autograd.grad
    df_dx_autograd = torch.autograd.grad(f_chain, x_chain)[0]

    # Аналитическое вычисление по цепному правилу: d/dx [sin(u)] = cos(u) * u', где u = x^2 + 1
    # df/dx = cos(x^2 + 1) * 2x
    df_dx_ana = torch.cos(x_chain**2 + 1) * 2 * x_chain

    print(f"f(x) = sin(x^2 + 1) при x = {x_chain.item()}")
    print(f"df/dx (torch.autograd.grad): {df_dx_autograd.item():.6f}")
    print(f"df/dx (аналитически):         {df_dx_ana.item():.6f}")

    assert torch.isclose(df_dx_autograd, df_dx_ana), "Градиенты по цепному правилу не совпадают!"

if __name__ == "__main__":
    homework_autograd()