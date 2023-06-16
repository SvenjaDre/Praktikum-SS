import matplotlib.pyplot as plt
import numpy as np
from uncertainties import ufloat
import uncertainties.unumpy as unp 
######## P und w Rechnung #####

def p(T):
    return 5.5*10**7*unp.exp(-6876/(T+273.15))

t = np.array([27.7, 148, 196, 198.4, 170, 173, 161, 154])
T = unp.uarray(t, 1)

print(p(T))

def w(p_T):
    return 0.0029/p_T

print('Wellenlänge')
print(w(p(T)))