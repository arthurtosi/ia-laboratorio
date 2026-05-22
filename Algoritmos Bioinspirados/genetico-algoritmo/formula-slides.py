import numpy as np

DIMENSIONS      = 30
POPULATION_SIZE = 1000
EPOCHS          = 500
CROSSOVER_RATE  = 0.85
MUTATION_RATE   = 0.1
LOWER_BOUND     = -32.0
UPPER_BOUND     =  32.0

def ackley(individual: np.ndarray) -> float:
    n       = len(individual)
    sum_sq  = np.sum(individual ** 2)
    sum_cos = np.sum(np.cos(2 * np.pi * individual))
    return -20 * np.exp(-0.2 * np.sqrt(sum_sq / n)) - np.exp(sum_cos / n) + 20 + np.e

def select(population: np.ndarray, fitnesses: np.ndarray) -> np.ndarray:
    new_pop = np.empty_like(population)
    for i in range(POPULATION_SIZE):
        idx_1 = np.random.randint(0, POPULATION_SIZE)
        idx_2 = np.random.randint(0, POPULATION_SIZE)
        if fitnesses[idx_1] < fitnesses[idx_2]:  # minimize: smaller is better
            new_pop[i] = population[idx_1]
        else:
            new_pop[i] = population[idx_2]
    return new_pop

def crossover(population: np.ndarray):
    for _ in range(POPULATION_SIZE):
        if np.random.rand() > CROSSOVER_RATE:
            continue
        cut_point = np.random.randint(1, DIMENSIONS)
        idx_1     = np.random.randint(0, POPULATION_SIZE)
        idx_2     = np.random.randint(0, POPULATION_SIZE)
        parent_1  = np.copy(population[idx_1])
        parent_2  = np.copy(population[idx_2])
        population[idx_1] = np.concatenate([parent_1[:cut_point], parent_2[cut_point:]])
        population[idx_2] = np.concatenate([parent_2[:cut_point], parent_1[cut_point:]])

def mutation(population: np.ndarray):
    mutation_mask = np.random.rand(POPULATION_SIZE, DIMENSIONS) <= MUTATION_RATE
    noise = np.random.normal(loc=0, scale=1.0, size=population.shape)
    population[mutation_mask] += noise[mutation_mask]
    np.clip(population, LOWER_BOUND, UPPER_BOUND, out=population)

if __name__ == "__main__":
    population = np.random.uniform(LOWER_BOUND, UPPER_BOUND, size=(POPULATION_SIZE, DIMENSIONS))

    for epoch in range(EPOCHS):
        fitnesses  = np.array([ackley(ind) for ind in population])
        population = select(population, fitnesses)
        crossover(population)
        mutation(population)

    fitnesses       = np.array([ackley(ind) for ind in population])
    best_index      = np.argmin(fitnesses)
    best_individual = population[best_index]
    best_fitness    = fitnesses[best_index]

    print(f"Best fitness : {best_fitness:.6f}")
    print(f"Best solution: {np.round(best_individual, 4)}")
