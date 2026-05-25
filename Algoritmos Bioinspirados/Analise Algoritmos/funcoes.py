import numpy as np


def schaffer_f2(individuo: np.ndarray) -> float:
    x1, x2 = individuo[0], individuo[1]

    numerador = np.sin(np.sqrt(x1**2 + x2**2))**2 - 0.5
    denominador = (1 + 0.001 * (x1**2 + x2**2))**2

    return 0.5 + numerador / denominador


def ackley(individuo: np.ndarray) -> float:
    n = len(individuo)

    sum_sq = np.sum(individuo**2)
    sum_cos = np.sum(np.cos(2 * np.pi * individuo))

    return (
        -20 * np.exp(-0.2 * np.sqrt(sum_sq / n))
        - np.exp(sum_cos / n)
        + 20
        + np.e
    )


def rosenbrock(individuo: np.ndarray) -> float:
    x_atual = individuo[:-1]
    x_proximo = individuo[1:]

    return np.sum(100 * (x_atual**2 - x_proximo) ** 2 + (x_atual - 1) ** 2)


def rastrigin(individuo: np.ndarray) -> float:
    n = len(individuo)

    return np.sum(individuo**2 - 10 * np.cos(2 * np.pi * individuo) + 10)


def griewank(individuo: np.ndarray) -> float:
    n = len(individuo)

    soma = np.sum(individuo**2) / 4000
    produto = np.prod(np.cos(individuo / np.sqrt(np.arange(1, n + 1))))

    return soma - produto + 1