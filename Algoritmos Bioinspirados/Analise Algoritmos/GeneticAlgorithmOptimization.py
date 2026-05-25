import numpy as np
from tqdm import tqdm


class GA:
    def __init__(
        self,
        bounds,
        fitness_func,
        dimensao,
        n_individuos=1000,
        taxa_cruzamento=0.85,
        taxa_mutacao=0.1,
    ):
        self._bounds = bounds
        self._fitness = fitness_func
        self._dimensao = dimensao
        self._n_individuos = n_individuos
        self._taxa_cruzamento = taxa_cruzamento
        self._taxa_mutacao = taxa_mutacao

        self._lower_bound = bounds[0]
        self._upper_bound = bounds[1]
        self._populacao = self._cria_populacao()
        self._fitnesses = self._calcula_fitnesses()

        melhor_idx = np.argmin(self._fitnesses)
        self._melhor_fit = self._fitnesses[melhor_idx]
        self._melhor_individuo = np.copy(self._populacao[melhor_idx])

    def _cria_populacao(self):
        return np.random.randint(
            low=self._lower_bound,
            high=self._upper_bound + 1,
            size=(self._n_individuos, self._dimensao),
        )

    def _calcula_fitnesses(self):
        return np.array([self._fitness(individuo) for individuo in self._populacao])

    def _atualiza_melhor(self):
        melhor_idx = np.argmin(self._fitnesses)

        if self._fitnesses[melhor_idx] < self._melhor_fit:
            self._melhor_fit = self._fitnesses[melhor_idx]
            self._melhor_individuo = np.copy(self._populacao[melhor_idx])

    def select(self):
        new_pop = np.full(
            shape=(self._n_individuos, self._dimensao),
            fill_value=-1,
            dtype=int,
        )

        for i in range(self._n_individuos):
            idx_1 = np.random.randint(low=0, high=self._n_individuos)
            fitness_1 = self._fitnesses[idx_1]

            idx_2 = np.random.randint(low=0, high=self._n_individuos)
            fitness_2 = self._fitnesses[idx_2]

            if fitness_1 > fitness_2:
                new_pop[i] = np.copy(self._populacao[idx_2])
            else:
                new_pop[i] = np.copy(self._populacao[idx_1])

        self._populacao = new_pop

    def crossover(self):
        for _ in range(self._n_individuos):
            rand_val = np.random.rand()
            if rand_val > self._taxa_cruzamento:
                continue

            cut_point = np.random.randint(low=1, high=self._dimensao)
            idx_1 = np.random.randint(low=0, high=self._n_individuos)
            idx_2 = np.random.randint(low=0, high=self._n_individuos)

            parent_1 = np.copy(self._populacao[idx_1])
            parent_2 = np.copy(self._populacao[idx_2])

            self._populacao[idx_1] = np.concatenate(
                [parent_1[:cut_point], parent_2[cut_point:]]
            )
            self._populacao[idx_2] = np.concatenate(
                [parent_2[:cut_point], parent_1[cut_point:]]
            )

    def mutation(self):
        random_draw = np.random.rand(self._n_individuos, self._dimensao)
        mutation_mask = random_draw <= self._taxa_mutacao
        n_mutations = np.sum(mutation_mask)

        noise = np.random.choice([-1, 1], size=(n_mutations,))
        self._populacao[mutation_mask] += noise
        np.clip(
            self._populacao,
            self._lower_bound,
            self._upper_bound,
            out=self._populacao,
        )

    def get_melhor_individuo(self):
        return np.copy(self._melhor_individuo)

    def otimiza(self, epocas: int):
        historico = [self._melhor_fit]

        with tqdm(total=epocas) as pbar:
            for _ in range(epocas):
                self._fitnesses = self._calcula_fitnesses()

                self.select()
                self.crossover()
                self.mutation()

                self._fitnesses = self._calcula_fitnesses()
                self._atualiza_melhor()
                historico.append(self._melhor_fit)

                pbar.update(1)

        return self._melhor_fit, historico


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    def fitness(individuo: np.ndarray) -> float:
        return float(np.sum(individuo ** 2))

    ga = GA(
        bounds=(-10, 10),
        fitness_func=fitness,
        dimensao=30,
        n_individuos=100,
    )
    melhor, historico = ga.otimiza(100)

    print(f"Melhor Fitness: {melhor}")
    print(f"Melhor Individuo: {ga.get_melhor_individuo()}")

    plt.plot(historico)
    plt.grid(True)
    plt.title("Evolucao do Fitness")
    plt.xlabel("Epocas")
    plt.ylabel("Fitness")
    plt.show()
