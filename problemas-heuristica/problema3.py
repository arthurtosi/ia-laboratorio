import numpy as np

DIMENSAO = 30
MAX_INTERVALO = 32
MIN_INTERVALO = -32
MAX_TEMP = 100
MIN_TEMP = 1e-3
ALPHA = 0.99

def func(X):
    n = len(X) # Será de tamanho N
    return -20 * np.exp(-0.2 * np.sqrt(np.sum(X**2) / n)) - np.exp(np.sum(np.cos(2 * np.pi * X)) / n) + 20 + np.e

def verify_constraints(X) -> bool:
    return np.all((MIN_INTERVALO <= X) & (X <= MAX_INTERVALO))

def get_neighbor(X, Temp):
    while True:
        new = np.copy(X)
        index = np.random.randint(0, DIMENSAO)
        change = np.random.uniform(-1,1) * Temp / MAX_TEMP
        new[index] += change
        
        # Tinha feito: 
        # change = np.random.uniform(-1,1, DIMENSAO) * Temp / MAX_TEMP
        # new += change
        
        if (verify_constraints(new)):
            return new

def annealing(restarts = 10):
    best_global = None
    
    for _ in range(restarts):
        current = np.random.uniform(MIN_INTERVALO, MAX_INTERVALO, DIMENSAO)
        best = current
        T = MAX_TEMP
        
        while T > MIN_TEMP:
            candidate = get_neighbor(current, T)
            
            candidate_func = func(candidate)
            current_func = func(current)
            delta = current_func - candidate_func
            
            if (delta > 0):
                current = np.copy(candidate)
            else:
                if (np.random.uniform(0,100) < np.exp(delta / T)):
                    current = np.copy(candidate)
            
            if (func(best) > current_func):
                best = np.copy(current)
                
            T *= ALPHA
            
        if (best_global is None or func(best_global) > func(best)):
            best_global = np.copy(best)
        
    return best_global

if __name__ == "__main__":
    result = annealing()
    print(f"Resultado = {result}")
    print(f"f(x) = {func(result)}")
    print(f"Dentro do intervalo = {verify_constraints(result)}")
    print(f"Menor valor do vetor = {np.min(result)}")
    print(f"Maior valor do vetor = {np.max(result)}")
    
    
    