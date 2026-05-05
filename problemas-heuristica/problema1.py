import random
import numpy as np
import math


def return_revenue(costs) -> float:
    cG, cI, cL, cY = costs
    rG = 50*cG - 1.2*cG**2
    rI = 45*cI - 1.0*cI**2
    rL = 40*cL - 0.8*cL**2
    rY = 55*cY - 1.5*cY**2
    return rG + rI + rL + rY


def verify_constraints(costs: np.ndarray) -> bool:
    if (costs.min() < 0): return False

    cG, cI, cL, cY = costs
    fullCost = cG + cI + cL + cY

    if (fullCost > 50 or fullCost < 0): return False
    if (cL + cY > 25): return False

    eG = cG * 2
    eI = cI * 1
    eL = cL * 3
    eY = cY * 2

    if (eG + eI + eY + eL <= 80):
        return True
    return False

def get_neighbor(costs: np.ndarray) -> np.ndarray:
    while 1:
        new = np.copy(costs)

        r = random.randint(0,3) # Elemento que será alterado
        change = random.choice([-1,1])

        new[r] += change

        if (verify_constraints(new)): return new
    
def annealing():
    current = np.array([0,0,0,0])
    best = current
    Temp = 100.0
    T_min = 0.01
    alpha = 0.99

    while T_min < Temp:
        candidate = get_neighbor(current)
        a = return_revenue(candidate)
        b  = return_revenue(current)
        delta = a - b 

        if (delta > 0):
            current = candidate
        else:
            prob = math.exp(delta / Temp)

            if random.random() < prob:
                current = candidate
        
        if (return_revenue(current) - return_revenue(best)) > 0:
            best = current
        
        Temp *= alpha
    
    return best

if __name__ == "__main__":
    result = annealing()
    print(f"Google ads = {result[0]* 1000}")
    print(f"Instagram ads = {result[1]* 1000}")
    print(f"Linkedin ads = {result[2]* 1000}")
    print(f"YouTube ads = {result[3]* 1000}")
    print (f"TOTAL = {return_revenue(result)*1000}")