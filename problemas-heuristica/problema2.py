import numpy as np
import matplotlib.pyplot as plt


MIN_X = -1
MAX_X = 2


def funcao(x):
    # np.sin também funciona com arrays então serve para o plot e para um x só
    return x * np.sin(10 * np.pi * x) + 1


def verify_constraints(x) -> bool:
    return MIN_X <= x <= MAX_X


def get_neighbor(x, Temp):
    while True:
        # Ruído maior no começo e menor no fim, acompanhando a temperatura
        change = np.random.uniform(-1, 1) * Temp / 100
        new = x + change

        if verify_constraints(new):
            return new


def annealing(restarts=10):
    global_best = None

    for _ in range(restarts):
        current = np.random.uniform(MIN_X, MAX_X)
        best = current
        Temp = 100.0
        T_min = 0.01
        alpha = 0.99

        while T_min < Temp:
            candidate = get_neighbor(current, Temp)
            a = funcao(candidate)
            b = funcao(current)
            delta = a - b

            if delta > 0:
                current = candidate
            else:
                prob = np.exp(delta / Temp)

                if np.random.random() < prob:
                    current = candidate

            if funcao(current) > funcao(best):
                best = current

            Temp *= alpha

        if global_best is None or funcao(best) > funcao(global_best):
            global_best = best

    return global_best


def plotar_funcao(best):
    # 3000 é a quantidade de pontos no intervalo
    x = np.linspace(MIN_X, MAX_X, 3000)
    y = funcao(x)

    plt.plot(x, y)
    plt.scatter(best, funcao(best), color="red")
    plt.show()


if __name__ == "__main__":
    result = annealing()
    print(f"x = {result}")
    print(f"f(x) = {funcao(result)}")
    plotar_funcao(result)
