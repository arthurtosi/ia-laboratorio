import numpy as np

DIMENSAO = 30
N_INDIVIDUOS = 1000
EPOCAS = 500
TAXA_CRUZAMENTO = 0.5 #Chance de selecionar o gene mutado
F = 0.4

def func(individuo: np.ndarray) -> float:
    n = len(individuo)
    sum_sq = np.sum(individuo ** 2)
    sum_cos = np.sum(np.cos(2 * np.pi * individuo))
    return -20 * np.exp(-0.2 * np.sqrt(sum_sq / n)) - np.exp(sum_cos / n) + 20 + np.e

def mutacao(populacao: np.ndarray):
    new_pop = np.zeros(shape=(N_INDIVIDUOS, DIMENSAO))
    
    for i in range(N_INDIVIDUOS):
        candidatos = [j for j in range(0,N_INDIVIDUOS) if j != i]
        idx_1, idx_2, idx_3 = np.random.choice(candidatos, size=3, replace=False)
        new_pop[i] = populacao[idx_1] + F*(populacao[idx_2] - populacao[idx_3])
        
    
    return np.clip(new_pop, -32, 32)

def crossover(populacao_normal, populacao_mutada):
    populacao_teste = np.copy(populacao_normal)
    probabilidades = np.random.rand(N_INDIVIDUOS, DIMENSAO)
    mascara = probabilidades <= TAXA_CRUZAMENTO
    
    populacao_teste[mascara] = populacao_mutada[mascara]
    
    return populacao_teste

def selecao(pop_normal, pop_teste):
    new_pop = np.empty_like(pop_normal)
    fitness_pop_normal = np.apply_along_axis(func, axis=1, arr=pop_normal)
    fitness_pop_teste = np.apply_along_axis(func, axis=1, arr=pop_teste)
    
    for i in range(N_INDIVIDUOS):        
        if (fitness_pop_normal[i] > fitness_pop_teste[i]):
            new_pop[i] = np.copy(pop_teste[i])
        else:
            new_pop[i] = np.copy(pop_normal[i])
    return new_pop
            

def differential_evolution():
    populacao = np.random.uniform(low=-32, high=32, size=(N_INDIVIDUOS, DIMENSAO))
    
    for _ in range(EPOCAS):
        # MUTACAO
        pop_mutada = mutacao(populacao)
        
        # CROSSOVER
        pop_teste = crossover(populacao, pop_mutada)
        
        # SELEÇÃO
        populacao = selecao(populacao, pop_teste)
    
    # resultados = [func(idx) for idx in populacao]
    resultados = np.apply_along_axis(func, 1, populacao)
    melhor_idx = np.argmin(resultados)
    melhor_individuo = populacao[melhor_idx]
    
    return melhor_individuo
        
        
        
if __name__ == "__main__":
    melhor = differential_evolution()
    print(f"Indivíduo: {melhor}\n\n Resultado = {func(melhor):.6f}")