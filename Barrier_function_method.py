from typing import List
import numpy as np
import math


def f_func(f: List[float], x: List[float]) -> float:
    return f[0] * x[0] ** 2 + f[1] * x[1] ** 2 + f[2]


def g_func(g: List[float], x: List[float]) -> float:
    return g[0] * x[0] + g[1] * x[1] + g[2]


def F(f: List[float], g: List[float], r, x: List[float]) -> float:
    constraint = g_func(g, x)
    if constraint >= -1e-10:
        return float('inf')
    return f_func(f, x) - r * np.log(-constraint)


def gradient(f: List[float], g: List[float], r: float, x: List[float]) -> List[float]:
    constraint = g_func(g, x)
    if constraint >= -1e-10:
        return [float('inf'), float('inf')]
    return [
        2 * f[0] * x[0] - r * g[0] / constraint,
        2 * f[1] * x[1] - r * g[1] / constraint
    ]


def gradient_descent(f_coef: List[float], g_coef: List[float], r: float, x0: List[float]) -> List[float]:
    e = 0.001
    x = x0.copy()
    prev_x = x.copy()

    print(f"\nНачальная точка: x = ({x[0]}, {x[1]})")
    print(f"Начальное значение функции: f(x) = {F(f_coef, g_coef, r, x):.3f}\n")

    for _ in range(1000):
        if g_func(g_coef, x) >= -1e-10:
            break
        grad = gradient(f_coef, g_coef, r, x)
        grad_norm = math.sqrt(grad[0] ** 2 + grad[1] ** 2)

        if grad_norm < e:
            break

        t = 0.1
        for _ in range(20):
            new_x = [x[0] - t * grad[0], x[1] - t * grad[1]]
            if g_func(g_coef, new_x) < -1e-10 and F(f_coef, g_coef, r, new_x) < F(f_coef, g_coef, r, x):
                break
            t *= 0.5
        else:
            break
        x = new_x

        delta_x = math.sqrt((new_x[0] - x[0]) ** 2 + (new_x[1] - x[1]) ** 2)
        delta_f = abs(F(f_coef, g_coef, r, new_x) - F(f_coef, g_coef, r, x))

        if delta_x < e and delta_f < e:
            delta_x_prev = math.sqrt((x[0] - prev_x[0]) ** 2 + (x[1] - prev_x[1]) ** 2)
            delta_f_prev = abs(F(f_coef, g_coef, r, x) - F(f_coef, g_coef, r, prev_x))

            if delta_x_prev < e and delta_f_prev < e:
                print(f"\nУсловие остановки по малым изменениям выполнено")
                break

    return x


def find_feasible_point(g_coef: List[float]) -> List[float]:
    strategies = [
        lambda: [-5.0, -(g_coef[2] - 5 * g_coef[0]) / g_coef[1] if g_coef[1] != 0 else -5.0],
        lambda: [-(g_coef[2] - 5 * g_coef[1]) / g_coef[0] if g_coef[0] != 0 else -5.0, -5.0],
        lambda: [-2.0, -2.0],
        lambda: [0.0, -(g_coef[2] + 0.1) / g_coef[1]] if g_coef[1] != 0 else [-(g_coef[2] + 0.1) / g_coef[0], 0.0]
    ]
    for strategy in strategies:
        try:
            x = strategy()
            if g_func(g_coef, x) < -1e-10:
                return x
        except:
            continue

    raise ValueError("Не удалось найти допустимую точку")


def Barrier_function_method() -> None:
    f_coef = [8, 1, 0]
    g_coef = [3, 1, 16]
    r = 1
    C = 5
    e = 0.05
    x = [-5, -5]

    if g_func(g_coef, x) >= -1e-10:
        try:
            x = find_feasible_point(g_coef)

        except ValueError as e:
            print(f"Ошибка: {e}")
            return

    print(f"Начальная точка: {x}, g(x) = {g_func(g_coef, x):.3f}")
    k = 0
    for _ in range(100):
        print(f"\nИтерация: {k}")
        print(f"r = {r:3f}")
        print(f"Текущая точка: ({x[0]:.3f}, {x[1]:.3f})")
        print(f"f(x) = {f_func(f_coef, x):.3f}")
        print(f"g(x) = {g_func(g_coef, x):.3f}")

        x = gradient_descent(f_coef, g_coef, r, x)

        barrier_value = abs(- r * np.log(-g_func(g_coef, x)))
        print(f"P(x, r) = {barrier_value:.3f}")

        if barrier_value <= e:
            print("\nУсловие остановки достигнуто: |P(x, r)| <= ε")
            break

        r /= C
        k += 1

    print("\nРезультаты:")
    print(f"Найденная точка: ({x[0]:.3f}, {x[1]:.3f})")
    print(f"Значение целефой функции f: {f_coef[0] * x[0] ** 2 + f_coef[1] * x[1] ** 2 + f_coef[2]:.3f}")
    print(f"Значение ограничения g: {abs(g_coef[0] * x[0] + g_coef[1] * x[1] + g_coef[2]):.3f}")
    print(f"Количество итераций: {k + 1}")


if __name__ == "__main__":
    Barrier_function_method()