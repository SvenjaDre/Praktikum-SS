import numpy as np
import scipy.constants as const

def Bragg(E):
    d=201.4e-12
    sin=const.c*const.h/(2*d*E*const.e)
    return np.arcsin(sin)

def sigma(Z,E):
    Ry=13.6
    alp=const.alpha
    return Z-np.sqrt(E/Ry-(alp**2*Z**4)/4)

ECu=np.array([8e3,8.9e3])
print(Bragg(ECu)*360/(2*np.pi))

En=np.array([9.65,11.11,13.48,15.2,16.12,18.01])*1e3
Z=np.array([30,32,35,37,38,40])
print(Bragg(En)*360/(2*np.pi))
print(sigma(Z,En))