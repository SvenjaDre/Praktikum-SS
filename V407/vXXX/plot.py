import matplotlib.pyplot as plt
import numpy as np
from uncertainties import ufloat

Daten = np.loadtxt('senkr.csv', delimiter=',')
daten = np.loadtxt('parallel.csv', delimiter=',')

I_dunkel = 1.4*10**(-9)

I_0 = 180*10**(-6)-I_dunkel
a_p = daten[:,0]
I_p = daten[:,1]
I_p = I_p*10**(-6) - I_dunkel

a_s = Daten[:,0]
I_s = Daten[:,1]
I_s = I_s*10**(-6) - I_dunkel



E_p = np.sqrt(I_p/I_0)
E_s = np.sqrt(I_s/I_0)


##### n Senkrecht ##############
#print('Brechungsindex Senkrecht polarisiert')
#print()
#def n_S (a, E):
#    return np.sqrt((1 + E**2 + 2*E*np.cos(2*a*np.pi/180))/(1 - 2*E +E**2))
#
#n_s = n_S(a_s, E_s) 
#print(n_s)
#print()
#n_smittel = np.mean(n_s)
#print('Mittelwert senk: ',n_smittel)
#n_err = np.std(n_s)
#print('standardabweicuhung', np.std(n_s))
#n_errr = ufloat(np.mean(n_s), np.std(n_s))
#print()

############################## n parallel ##########################
print('Brechungsindex parallel polarisiert')
print()
def n_P(a, E):
    b = ((E+1)/(E-1))**2
    return np.sqrt(b/(2*np.cos(a)**2) + np.sqrt(b**2/(4*np.cos(a)**4) - b*np.tan(a)**2))

n_p = n_P(a_p, E_p)
print(n_p)
print()
n_pmittel = np.mean(n_p)
print('Mittelwert parra: ', n_pmittel)
n_perr = np.std(n_p)
print('Standardabweichung', n_perr)

################# Plot #####################

#def theorie_senk(a, n):
#    return (np.sqrt(n^2 - np.sin(a*np.pi/180)**2) - np.cos(a*np.pi/180))**2/(n^2 -1)

def KurveS(a, n):
    return -(np.cos(a*np.pi/180) - np.sqrt(n**2-np.sin(a*np.pi/180)**2)) / (np.cos(a*np.pi/180) + np.sqrt(n**2-np.sin(a*np.pi/180)**2))

def KurveP(a, n):
    return (n**2*np.cos(a*np.pi/180) - np.sqrt(n**2-np.sin(a*np.pi/180)**2)) / (n**2*np.cos(a*np.pi/180) + np.sqrt(n**2-np.sin(a*np.pi/180)**2))  

alpha_B = 80.75
alpha = np.linspace(0, 90, 1000)
alpha_1 = np.linspace(0, alpha_B, 1000)
alpha_2 = np.linspace(alpha_B, 90, 1000)

plt.plot(alpha, KurveS(alpha, n_smittel), color = "blue", label = "Theoriekurve senkrecht polarisiert")
plt.plot(alpha_1, KurveP(alpha_1, n_pmittel), color = "purple", label = "Theoriekurve parallel polarisiert")
plt.plot(alpha_2, -KurveP(alpha_2, n_pmittel), color = "purple")

plt.plot(a_p, E_p, 'rx', label = 'Messwerte parallel polarisiert')
plt.plot(a_s, E_s, 'gx', label = 'Messwerte senkrecht polarisiert')
plt.xlabel(r"$\alpha / °$")
plt.ylabel(r"$\sqrt{I_r / I_0}$")
plt.grid(True)
plt.legend()

plt.savefig('build/plot.pdf')

brew = 70
n = np.tan(brew*np.pi/180)
print(n)
