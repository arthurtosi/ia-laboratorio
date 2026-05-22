import numpy as np

MANNING_COEFFICIENT = 0.013
SLOPE = 0.005
POPULATION_SIZE = 1000
N_PIPES = 10
EPOCHS  = 50
CROSSOVER_RATE  = 0.85
MUTATION_RATE  = 0.1

LENGTHS = np.array([20, 54, 98, 120, 34, 12, 88, 122, 33, 40])
FLOWS   = np.array([i * 2.5 for i in range(1, 11)])

def calc_qmax(diameter: float) -> float: # Usado só
    diameter /= 1000
    radius = diameter / 2.0
    A = np.pi * np.square(radius)
    Rh = diameter / 4.0
    Q_max = 1/MANNING_COEFFICIENT * A * np.power(Rh, 2/3) * np.sqrt(SLOPE)
    Q_max *= 1000  # m3/s -> L/s
    return Q_max

def verify_constraints(section: int, pipe_idx: int) -> bool:
    return FLOWS[section] <= 0.75 * pipes["Q_max"][pipe_idx]

def fitness(individual: np.ndarray) -> float:
    cost = 0
    for i in range(N_PIPES):
        cost += LENGTHS[i] * pipes["unit_costs"][individual[i]]
        if not verify_constraints(i, individual[i]):
            cost += 1e6
    return cost

def select(population: np.ndarray, fitnesses: np.ndarray) -> np.ndarray:
    new_pop = np.full(shape=(POPULATION_SIZE, N_PIPES), fill_value=-1)
    for i in range(POPULATION_SIZE):
        idx_1 = np.random.randint(low=0, high=POPULATION_SIZE)
        fitness_1 = fitnesses[idx_1]

        idx_2    = np.random.randint(low=0, high=POPULATION_SIZE) 
        fitness_2 = fitnesses[idx_2]

        if fitness_1 > fitness_2:
            new_pop[i] = np.copy(population[idx_2])
        else:
            new_pop[i] = np.copy(population[idx_1])
    return new_pop

def crossover(population: np.ndarray):
    for _ in range(POPULATION_SIZE):
        rand_val = np.random.rand()
        if rand_val > CROSSOVER_RATE:
            continue

        cut_point = np.random.randint(low=1, high=N_PIPES)
        idx_1 = np.random.randint(low=0, high=POPULATION_SIZE)
        idx_2 = np.random.randint(low=0, high=POPULATION_SIZE)

        parent_1 = np.copy(population[idx_1])
        parent_2 = np.copy(population[idx_2])

        population[idx_1] = np.concatenate([parent_1[:cut_point], parent_2[cut_point:]])
        population[idx_2] = np.concatenate([parent_2[:cut_point], parent_1[cut_point:]])

def mutation(population: np.ndarray):
    random_draw = np.random.rand(POPULATION_SIZE, N_PIPES)
    mutation_mask = random_draw <= MUTATION_RATE
    n_mutations = np.sum(mutation_mask)

    noise = np.random.choice([-1, 1], size=(n_mutations,))
    population[mutation_mask] += noise
    np.clip(population, 0, 4, out=population) #out serve para mudar population como referência (por padrão o .clip retorna um np.array)

if __name__ == "__main__":
    population = np.random.randint(low=0, high=5, size=(POPULATION_SIZE, N_PIPES)) #high exclusivo

    pipes = {
        "unit_costs": [65.0, 98.0, 150.0, 210.0, 340.0],
        "diameters": [150.0, 200.0, 250.0, 300.0, 400.0],
        "Q_max": [calc_qmax(d) for d in [150.0, 200.0, 250.0, 300.0, 400.0]]
    }

    for _ in range(EPOCHS):
        fitnesses = np.array([fitness(ind) for ind in population])

        # Select parents
        population = select(population, fitnesses)

        # Crossover
        crossover(population)

        # Mutation
        mutation(population)

    fitnesses = np.array([fitness(ind) for ind in population])
    best_index = np.argmin(fitnesses)
    best_individual = population[best_index]

    print(f"\n{'Section':<8} {'Length':>10} {'Type':>7} {'Diameter':>10} {'R$/m':>8} {'Cost':>9}\n")
    total_cost = 0
    for i, pipe_idx in enumerate(best_individual):
        length = LENGTHS[i]
        diameter = pipes["diameters"][pipe_idx]
        unit_cost = pipes["unit_costs"][pipe_idx]
        cost = length * unit_cost
        total_cost += cost
        print(f"{i+1:<8} {length:>10}m {pipe_idx:>7} {diameter:>10}mm {unit_cost:>8} R$ {cost:>9}\n")
    print(f"{'Total cost':>44} R$ {total_cost:>9.2f}")
