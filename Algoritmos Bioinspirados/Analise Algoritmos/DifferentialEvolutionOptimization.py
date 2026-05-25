import numpy as np
from tqdm import tqdm

def fitness(individuo: np.ndarray) -> float:
        n = len(individuo)
        sum_sq = np.sum(individuo ** 2)
        sum_cos = np.sum(np.cos(2 * np.pi * individuo))
        return -20 * np.exp(-0.2 * np.sqrt(sum_sq / n)) - np.exp(sum_cos / n) + 20 + np.e

class DE:
    def __init__(
        self,
        bounds,
        fitness_func,
        dimensao=30,
        n_individuos=50,
        taxa_cruzamento=0.4,
        f=0.5,
    ):
        if n_individuos < 4:
            raise ValueError("Differential Evolution precisa de pelo menos 4 individuos.")

        self._bounds = bounds
        self._fitness = fitness_func
        self._dimensao = dimensao
        self._n_individuos = n_individuos
        self._taxa_cruzamento = taxa_cruzamento
        self._f = f

        self._lower_bound = bounds[0]
        self._upper_bound = bounds[1]
        self._populacao = np.random.uniform(
            low=self._lower_bound,
            high=self._upper_bound,
            size=(self._n_individuos, self._dimensao),
        )
        self._fitnesses = self._calcula_fitnesses(self._populacao)

        melhor_idx = np.argmin(self._fitnesses)
        self._melhor_fit = self._fitnesses[melhor_idx]
        self._melhor_individuo = np.copy(self._populacao[melhor_idx])

    def _calcula_fitnesses(self, populacao):
        return np.array([self._fitness(individuo) for individuo in populacao])

    def _atualiza_melhor(self):
        melhor_idx = np.argmin(self._fitnesses)

        if self._fitnesses[melhor_idx] < self._melhor_fit:
            self._melhor_fit = self._fitnesses[melhor_idx]
            self._melhor_individuo = np.copy(self._populacao[melhor_idx])

    def mutacao(self):
        new_pop = np.zeros(shape=(self._n_individuos, self._dimensao))

        for i in range(self._n_individuos):
            candidatos = [j for j in range(0, self._n_individuos) if j != i]
            idx_1, idx_2, idx_3 = np.random.choice(candidatos, size=3, replace=False)
            new_pop[i] = self._populacao[idx_1] + self._f * (
                self._populacao[idx_2] - self._populacao[idx_3]
            )

        return np.clip(new_pop, self._lower_bound, self._upper_bound)

    def crossover(self, populacao_mutada):
        populacao_teste = np.copy(self._populacao)
        probabilidades = np.random.rand(self._n_individuos, self._dimensao)
        mascara = probabilidades <= self._taxa_cruzamento

        for i in range(self._n_individuos):
            j_rand = np.random.randint(0, self._dimensao)
            mascara[i, j_rand] = True

        populacao_teste[mascara] = populacao_mutada[mascara]

        return populacao_teste

    def selecao(self, populacao_teste):
        new_pop = np.empty_like(self._populacao)
        fitness_pop_normal = self._calcula_fitnesses(self._populacao)
        fitness_pop_teste = self._calcula_fitnesses(populacao_teste)

        for i in range(self._n_individuos):
            if fitness_pop_normal[i] > fitness_pop_teste[i]:
                new_pop[i] = np.copy(populacao_teste[i])
            else:
                new_pop[i] = np.copy(self._populacao[i])

        self._populacao = new_pop
        self._fitnesses = self._calcula_fitnesses(self._populacao)

    def get_melhor_individuo(self):
        return np.copy(self._melhor_individuo)

    def otimiza(self, epocas: int):
        historico_fitness = [self._melhor_fit]

        with tqdm(total=epocas, desc="Otimizando") as pbar:
            for _ in range(epocas):
                populacao_mutada = self.mutacao()
                populacao_teste = self.crossover(populacao_mutada)
                self.selecao(populacao_teste)

                self._atualiza_melhor()
                historico_fitness.append(self._melhor_fit)

                pbar.update(1)
                pbar.set_postfix({"Melhor Fitness": f"{self._melhor_fit:.5f}"})

        return self._melhor_fit, historico_fitness


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    de = DE(
        bounds=(-32, 32),
        fitness_func=fitness,
        dimensao=30,
        n_individuos=50,
    )
    melhor, historico = de.otimiza(1000)

    print(f"Melhor Fitness: {melhor}")
    print(f"Melhor Individuo: {de.get_melhor_individuo()}")

    plt.plot(historico)
    plt.title("Evolucao do Fitness")
    plt.xlabel("Epocas")
    plt.ylabel("Fitness")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.show()
