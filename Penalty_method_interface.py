from typing import List, Tuple, Dict
import math
import numpy as np


def f_func(f: List[float], x: List[float]) -> float:
    return f[0] * x[0] ** 2 + f[1] * x[1] ** 2 + f[2]


def g_func(g: List[float], x: List[float]) -> float:
    return g[0] * x[0] + g[1] * x[1] + g[2]


def F(f: List[float], g: List[float], r: float, x: List[float]) -> float:
    return f_func(f, x) + r / 2 * (g_func(g, x) ** 2)


def gradient(f: List[float], g: List[float], r: float, x: List[float]) -> List[float]:
    constraint = g_func(g, x)
    return [
        2 * f[0] * x[0] + r * g[0] * constraint,
        2 * f[1] * x[1] + r * g[1] * constraint
    ]

def Newton_method(f_coef: List[float], g_coef: List[float], r: float, x0: List[float]) -> List[float]:
    e = 0.001
    x = x0.copy()

    #while True:
    for _ in range(1000):
        grad = gradient(f_coef, g_coef, r, x)
        grad_norm = math.sqrt(grad[0]**2 + grad[1]**2)

        if grad_norm <= e:
            print(f"Остановка: норма градиента {grad_norm:.3f} <= {e}")
            return x

        Hesse_mat = np.array([
            [2 * f_coef[0] + r * g_coef[0] ** 2, r * g_coef[0] * g_coef[1]],
            [r * g_coef[0] * g_coef[1], 2 * f_coef[1] + r * g_coef[1] ** 2]])

        try:
            delta = -np.linalg.solve(Hesse_mat, grad)
            new_x = [x[0] + delta[0], x[1] + delta[1]]

            if F(f_coef, g_coef, r, new_x) < F(f_coef, g_coef, r, x):
                x = new_x
                continue

        except np.linalg.LinAlgError:
            pass

        t = 0.1
        for _ in range(20):
            new_x = [x[0] - t * grad[0], x[1] - t * grad[1]]
            if F(f_coef, g_coef, r, new_x) < F(f_coef, g_coef, r, x):
                x = new_x
                break
            t *= 0.5
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


def Penalty_method(f_coef: List[float], g_coef: List[float], r: float, C: float,
                   e: float, x: List[float]) -> Tuple[Dict, List[List[float]]]:
    points = [x.copy()]
    k = 0
    if g_func(g_coef, x) >= -1e-10:
        try:
            x = find_feasible_point(g_coef)

        except ValueError as e:
            print(f"Ошибка: {e}")
            return {
                'point': x,
                'value': f_func(f_coef, x),
                'iterations': k + 1
            }, points

    #while True:
    for _ in range(100):
        print(f"\nИтерация: {k}")
        print(f"r = {r}")
        print(f"Текущая точка: ({x[0]:.3f}, {x[1]:.3f})")

        x = Newton_method(f_coef, g_coef, r, x)

        penalty_value = r / 2 * (g_func(g_coef, x) ** 2)
        print(f"P(x, r) = {penalty_value:.3f}")
        if penalty_value <= e:
            print("\nУсловие остановки достигнуто: P(x, r) <= ε")
            print(f"Найденная точка: ({x[0]:.3f}, {x[1]:.3f})")
            print(
                f"Значение целефой функции f: {f_coef[0] * x[0] ** 2 + f_coef[1] * x[1] ** 2 + f_coef[2]:.3f}")
            print(f"Значение ограничения g: {g_coef[0] * x[0] + g_coef[1] * x[1] + g_coef[2]:.3f}")
            print(f"Количество итераций: {k + 1}")
            return {
                'point': x,
                'value': f_func(f_coef, x),
                'iterations': k + 1
            }, points

        r *= C
        print(f"r = {r}")
        k += 1