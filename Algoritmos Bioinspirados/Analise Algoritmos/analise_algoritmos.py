from time import perf_counter

import numpy as np


from ParticleSwarmOptimization import PSO
from GeneticAlgorithmOptimization import GA
from DifferentialEvolutionOptimization import DE

from funcoes import rosenbrock as fitness


LIM_INFERIOR = -50
LIM_SUPERIOR = 50
DIMENSAO = 30
N_INDIVIDUOS = 1000
EPOCAS = 50



def cria_algoritmos():
    bounds = (LIM_INFERIOR, LIM_SUPERIOR)

    return [
        (
            "PSO",
            PSO(
                bounds=bounds,
                fitness_func=fitness,
                coeficiente_inercia=0.4,
                dimensao=DIMENSAO,
                n_individuos=N_INDIVIDUOS,
            ),
        ),
        (
            "GA",
            GA(
                bounds=bounds,
                fitness_func=fitness,
                dimensao=DIMENSAO,
                n_individuos=N_INDIVIDUOS,
                taxa_cruzamento=0.85,
                taxa_mutacao=0.01,
            ),
        ),
        (
            "DE",
            DE(
                bounds=bounds,
                fitness_func=fitness,
                dimensao=DIMENSAO,
                n_individuos=N_INDIVIDUOS,
                taxa_cruzamento=0.4,
                f=0.5,
            ),
        ),
    ]


def executa_algoritmo(nome, algoritmo):
    inicio = perf_counter()
    melhor_fitness, historico = algoritmo.otimiza(EPOCAS)
    fim = perf_counter()

    return {
        "nome": nome,
        "melhor_fitness": melhor_fitness,
        "tempo": fim - inicio,
        "historico": historico,
    }


def imprime_resultados(resultados):
    print(f"\n{'Algoritmo':<12} {'Melhor Fitness':>18} {'Tempo (s)':>12}")
    print("-" * 45)

    for resultado in resultados:
        print(
            f"{resultado['nome']:<12} "
            f"{resultado['melhor_fitness']:>18.6f} "
            f"{resultado['tempo']:>12.4f}"
        )


def plota_historicos(resultados):
    import matplotlib.pyplot as plt

    for resultado in resultados:
        plt.plot(resultado["historico"], label=resultado["nome"])

    plt.grid(True)
    plt.title("Comparacao dos Algoritmos Bioinspirados")
    plt.xlabel("Epocas")
    plt.ylabel("Melhor Fitness")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    resultados = []
    for nome, algoritmo in cria_algoritmos():
        resultados.append(executa_algoritmo(nome, algoritmo))

    imprime_resultados(resultados)
    plota_historicos(resultados)
