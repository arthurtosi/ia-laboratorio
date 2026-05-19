import matplotlib.pyplot as plt
import numpy as np


MIN_X = -1
MAX_X = 2


def funcao(x):
    # np.sin funciona tanto para um número quanto para um array de valores.
    # Isso permite calcular f(x) para todos os pontos do gráfico de uma vez.
    return x * np.sin(10 * np.pi * x) + 1


def limitar_intervalo(x: float) -> float:
    # np.clip "prende" o valor no intervalo [MIN_X, MAX_X].
    # Se x < MIN_X, retorna MIN_X; se x > MAX_X, retorna MAX_X.
    return np.clip(x, MIN_X, MAX_X)


def gerar_vizinho(x: float, temperatura: float, rng: np.random.Generator) -> float:
    # np.maximum escolhe o maior entre os dois valores.
    # Assim a amplitude diminui com a temperatura, mas nunca fica menor que 0.001.
    amplitude = np.maximum(0.001, temperatura * (MAX_X - MIN_X))

    # rng.uniform sorteia um ruído contínuo entre -amplitude e +amplitude.
    # Esse ruído é somado ao x atual para gerar um vizinho.
    return limitar_intervalo(x + rng.uniform(-amplitude, amplitude))


def busca_local(x: float) -> float:
    melhor_x = x
    melhor_y = funcao(melhor_x)
    passo = 0.05

    while passo > 1e-8:
        # np.array cria os dois movimentos possíveis de uma vez:
        # testar um ponto à esquerda (-passo) e outro à direita (+passo).
        candidatos_x = limitar_intervalo(melhor_x + np.array([-passo, passo]))

        # Como funcao aceita arrays, candidatos_y terá os valores dos dois candidatos.
        candidatos_y = funcao(candidatos_x)

        # np.argmax retorna o índice do maior valor dentro do array.
        # Se candidatos_y[1] for maior, por exemplo, ele retorna 1.
        melhor_indice = np.argmax(candidatos_y)

        if candidatos_y[melhor_indice] > melhor_y:
            melhor_x = candidatos_x[melhor_indice]
            melhor_y = candidatos_y[melhor_indice]
        else:
            passo *= 0.5

    return melhor_x


def annealing(restarts: int = 50, seed: int | None = None) -> tuple[float, float]:
    # default_rng cria um gerador de números aleatórios do NumPy.
    # A seed permite repetir exatamente o mesmo experimento quando necessário.
    rng = np.random.default_rng(seed)
    melhor_global_x = None

    # -np.inf representa "menos infinito", garantindo que qualquer resultado real
    # encontrado será melhor que o valor inicial.
    melhor_global_y = -np.inf

    for _ in range(restarts):
        # Começa cada reinício em um ponto aleatório dentro do intervalo permitido.
        atual_x = rng.uniform(MIN_X, MAX_X)
        atual_y = funcao(atual_x)
        melhor_x = atual_x
        melhor_y = atual_y

        temperatura = 100.0
        temperatura_minima = 1e-2
        alpha = 0.99

        while temperatura > temperatura_minima:
            candidato_x = gerar_vizinho(atual_x, temperatura, rng)
            candidato_y = funcao(candidato_x)
            delta = candidato_y - atual_y

            # np.exp calcula e^(delta / temperatura).
            # Quanto maior a temperatura, maior a chance de aceitar uma piora.
            if delta > 0 or rng.random() < np.exp(delta / temperatura):
                atual_x = candidato_x
                atual_y = candidato_y

            if atual_y > melhor_y:
                melhor_x = atual_x
                melhor_y = atual_y

            temperatura *= alpha

        melhor_x = busca_local(melhor_x)
        melhor_y = funcao(melhor_x)

        if melhor_y > melhor_global_y:
            melhor_global_x = melhor_x
            melhor_global_y = melhor_y

    return melhor_global_x, melhor_global_y


def plotar_funcao(melhor_x: float | None = None):
    # np.linspace cria 3000 pontos igualmente espaçados entre MIN_X e MAX_X.
    # Isso é diferente de np.arange: aqui informamos a quantidade de pontos.
    x = np.linspace(MIN_X, MAX_X, 3000)

    # y é calculado de forma vetorizada: um valor de f(x) para cada ponto em x.
    y = funcao(x)

    plt.plot(x, y)
    plt.xlabel("x")
    plt.ylabel("f(x)")

    if melhor_x is not None:
        melhor_y = funcao(melhor_x)
        plt.scatter(melhor_x, melhor_y, color="red")
        plt.title(f"Máximo encontrado: x={melhor_x:.6f}, f(x)={melhor_y:.6f}")

    plt.show()


if __name__ == "__main__":
    x, y = annealing()
    print(f"x = {x:.10f}")
    print(f"f(x) = {y:.10f}")
    plotar_funcao(x)
