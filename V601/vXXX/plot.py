import matplotlib.pyplot as plt
import numpy as np
from uncertainties import ufloat
import uncertainties.unumpy as unp 
######## P und w Rechnung #####
a = 0.01
def p(T):
    return 5.5*10**7*unp.exp(-6876/(T+273.15))

t = np.array([27.7, 148, 154, 161, 170, 196 ])  #173,, 198.4
T = unp.uarray(t, 2)
#print(t+273.15)
print('Sättigungsdruck')
#print(p(T))

def w(p_T):
    return 0.0029/p_T

print('Wellenlänge')
#print(w(p(T)))
v = a/w(p(T)*10**2)
print(v)