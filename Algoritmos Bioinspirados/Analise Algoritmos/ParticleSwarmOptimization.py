import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

def fitness(individuo: np.ndarray) -> float:
    n = len(individuo)
    sum_sq = np.sum(individuo ** 2)
    sum_cos = np.sum(np.cos(2 * np.pi * individuo))
    return -20 * np.exp(-0.2 * np.sqrt(sum_sq / n)) - np.exp(sum_cos / n) + 20 + np.e

class PSO:
    def __init__(
        self,
        bounds, # Lower e upper bound do problema -> tupla()
        fitness_func,
        coeficiente_inercia=0.6,
        c1=2.05,
        c2=2.05,
        dimensao=30,
        n_individuos=1000, 
    ):
        self._coeficiente_inercia = coeficiente_inercia
        self._velocidades = np.random.uniform(-1,1, (n_individuos, dimensao))
        self._pos = np.random.uniform(bounds[0], bounds[1], (n_individuos, dimensao))
        
        self._pbest_pos = np.copy(self._pos) # (n_individuos, dimensao)
        # self.pbest_fit = np.apply_along_axis(fitness_func, axis=1, arr=self.pos)
        self._pbest_fit = np.array([fitness_func(p) for p in self._pos]) # (n_individuos,)
        
        self._current_fit = np.copy(self._pbest_fit) # (n_individuos,)
        
        global_idx = np.argmin(self._pbest_fit)
        self._gbest_fit = self._pbest_fit[global_idx] # Number
        self._gbest_pos = self._pbest_pos[global_idx] # (dimensao,)
        
        self._c1 = c1
        self._c2 = c2
        self._dimensao = dimensao
        self._n_individuos = n_individuos
        self._bounds = bounds
        self._fitness = fitness_func
    
    def get_melhores(self):
        melhores_mask = self._current_fit < self._pbest_fit
        self._pbest_pos[melhores_mask] = self._pos[melhores_mask]
        self._pbest_fit[melhores_mask] = self._current_fit[melhores_mask]
        
        global_idx = np.argmin(self._pbest_fit)
        self._gbest_fit = self._pbest_fit[global_idx]
        self._gbest_pos = self._pbest_pos[global_idx]
    
    def atualiza_velocidade(self):
        r1 = np.random.rand(self._n_individuos, self._dimensao)
        r2 = np.random.rand(self._n_individuos, self._dimensao)
        self._velocidades  = ( self._coeficiente_inercia * self._velocidades + 
                            self._c1 * r1 * (self._pbest_pos - self._pos) + 
                            self._c2 * r2 * (self._gbest_pos - self._pos) )
    
    def atualiza_posicoes(self):
        self._pos += self._velocidades
        np.clip(self._pos, self._bounds[0], self._bounds[1], out=self._pos)
        
    def otimiza(self, epocas: int):
        historico = []
        historico.append(self._gbest_fit)
        
        with tqdm (total=epocas) as pbar:
            for _ in range(epocas):
                self.get_melhores()
                self.atualiza_velocidade()
                self.atualiza_posicoes()
                self._current_fit = np.array([self._fitness(p) for p in self._pos])
                historico.append(self._gbest_fit)
                pbar.update(1)
        
        self.get_melhores()
        return self._gbest_fit, historico


LIM_INFERIOR = -32
LIM_SUPERIOR = 32
EPOCAS = 500
N_INDIVIDUOS = 100


if __name__ == "__main__":

    pso = PSO((LIM_INFERIOR, LIM_SUPERIOR), fitness, coeficiente_inercia=0.4, n_individuos=N_INDIVIDUOS)
    melhor, historico = pso.otimiza(EPOCAS)
    
    print(f"Melhor Fitness: {melhor}")
    
    plt.plot(historico)
    plt.grid(True)
    plt.title("Evolução do Fitness")
    plt.show()
