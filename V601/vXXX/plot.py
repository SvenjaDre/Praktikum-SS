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
#print('Sättigungsdruck')
#print(p(T))

def w(p_T):
    return 0.0029/p_T

#print('Wellenlänge')
#print(w(p(T)))
v = a/w(p(T)*10**2)
#print(v)


######### F-H-Kurven #######


T_198 = np.array([4.3, 4.3, 4.5, 4.5, 4.7, 4.5, 4.7, 4.7, 4.8, 4.3])
T_196 = np.array([4.5, 4.8, 4.6, 5.1, 5.1, 5.1, 5.3, 5.1, 5.1 ])
T_154 = np.array([4.8, 4.8, 5.1, 5.5])
T_161 = np.array([5.3, 5.6, 5.3, 5.8 ])
T_173 = np.array([4.5, 4.6, 4.5, 4.8, 4.8])
T_170 = np.array([4.3, 4.3, 4.3, 4.3, 4.8])

print('T = 198 :', np.mean(T_198))
print('T = 198 +- :', np.std(T_198))

print('T = 196 :', np.mean(T_196))
print('T = 196 +- :', np.std(T_196))

print('T = 154 :', np.mean(T_154))
print('T = 154 +- :', np.std(T_154))

print('T = 161 :', np.mean(T_161))
print('T = 161 +- :', np.std(T_161))

print('T = 173 :', np.mean(T_173))
print('T = 173 +- :', np.std(T_173))

print('T = 170 :', np.mean(T_170))
print('T = 170 +- :', np.std(T_170))

dU_b = ufloat(4.8, 0.2) 
def l(U):
    c = 2.99*10**8
    h = 6.626*10**-34
    return (c*h)/(U* 1.6*10**-19)

print('Wellenlänge: ', l(dU_b))

Daten = np.loadtxt('K27.csv', delimiter=',', skiprows=1)

U_B1 = Daten[:,0]
S_1 = Daten[:,1]

plt.plot(U_B1, S_1, 'rx', label='Messwerte')
plt.grid(True)
plt.xlabel(r'$U_B/ V$')
plt.ylabel(r'Steigung')
plt.legend()
plt.savefig('build/K27.pdf')
plt.clf()

Daten = np.loadtxt('K148.csv', delimiter=',', skiprows=1)

U_B2 = Daten[:,0]
S_2 = Daten[:,1]

plt.plot(U_B2, S_2, 'rx', label='Messwerte')
plt.grid(True)
plt.xlabel(r'$U_B/ V$')
plt.ylabel(r'Steigung')
plt.legend()
plt.savefig('build/K148.pdf')

UBmax = ufloat(8.5,0.5)

def K(U):
    return  (11 - U)

print('K=', K(UBmax))