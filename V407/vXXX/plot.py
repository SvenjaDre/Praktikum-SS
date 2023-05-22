import matplotlib.pyplot as plt
import numpy as np
from uncertainties import ufloat

Daten = np.loadtxt('senkr.csv', delimiter=',')
daten = np.loadtxt('parallel.csv', delimiter=',')

I_0 = 180*10**(-6)
I_dunkel = 1.4*10**(-9)

a_p = daten[:,0]
I_p = daten[:,1]
I_p = I_p*10**(-6) - I_dunkel

a_s = Daten[:,0]
I_s = Daten[:,1]
I_s = I_s*10**(-6) - I_dunkel



E_p = np.sqrt(I_p/I_0)
E_s = np.sqrt(I_s/I_0)


##### n Senkrecht ##############
print('Brechungsindex Senkrecht polarisiert')
def n_S (a, E):
    return np.sqrt((1 + E**2 + 2*E*np.cos(2*a*np.pi/180))/(1 - 2*E +E**2))

n_s = n_S(a_s, E_s) 
#print(n_s)
n_smittel = ufloat(np.mean(n_s), np.std(n_s))
print('Mittelwert senk: ',n_smittel)

######n parallel #####
print('Brechungsindex parallel polarisiert')
def n_P(a, E):
    b = ((E+1)/(E-1))**2
    return np.sqrt(b/(2*np.cos(a*np.pi/180)**2) + np.sqrt(b**2/(4*np.cos(a*np.pi/180)**4) - b*np.tan(a*np.pi/180)**2))

n_p = n_P(a_p, E_p)
#print(n_p)
n_pmittel = ufloat(np.mean(n_p), np.std(n_p))
print('Mittelwert parra: ', n_pmittel)

def theorie_para(a, n):
    return (np.sqrt{n^2})

plt.plot(a_p, E_p, 'rx', label = 'Messwerte parallel polarisiert')
plt.plot(a_s, E_s, 'gx', label = 'Messwerte senkrecht polarisiert')
plt.xlabel(r"$\alpha / °$")
plt.ylabel(r"$\sqrt{I_r / I_0}$")
plt.grid(True)
plt.legend()




plt.savefig('build/plot.pdf')
