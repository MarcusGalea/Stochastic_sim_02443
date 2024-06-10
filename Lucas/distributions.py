
#%%
import numpy.random as rnd
import numpy as np

def getExponential(parms):
    lam = parms[0]
    n = parms[1]
    Us = rnd.rand(n)
    return - np.log(Us) / lam 

def getUniform(a,b, n = 10000):
    Us = rnd.rand(n)
    Xs = a + (b - a) * Us
    return Xs

def getTrigonometric(n = 10000):
    Vs1 = np.zeros(n)
    Vs2 = Vs1
    for i in range(len(Vs1)):
        while True:
            V1 = getUniform(-1, 1, 1)
            V2 = getUniform(-1, 1, 1)
            Rsq = V1**2 + V2**2
            if  Rsq <= 1:
                break
        Vs1[i] = V1 / np.sqrt(Rsq)
        Vs2[i] = V2 / np.sqrt(Rsq)
    return Vs1 , Vs2 

def getNormal(mu = 0, sigma = 1, n = 10000):
    cos, sin = getTrigonometric(int(np.ceil(n / 2)))
    U1s = rnd.rand(int(np.ceil(n / 2)))
    cs = np.sqrt(-2 * np.log(U1s))
    Z1s = cs * cos
    Z2s = cs * sin
    Zs = np.append(Z1s, Z2s)
    Zs = Zs[:n-1]
    return Zs * sigma + mu
