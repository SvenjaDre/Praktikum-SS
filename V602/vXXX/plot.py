import matplotlib.pyplot as plt
import numpy as np
from uncertainties import ufloat
import uncertainties.unumpy as unp 
###### Bragg Bedingung ##############

Daten = np.loadtxt('ÜBraggbedingung.csv', delimiter=',',skiprows=1)

theta_1 = Daten[:,0]
I = Daten[:,1]
theta1_max = 28.2

plt.plot(theta_1, I, 'x', label='Messwerte')
plt.axvline(x=theta1_max, color='red', linestyle='-', label='Maximum')
plt.xlabel(r'$2 \theta /°$')
plt.ylabel(r'$Imps /s$')
plt.legend()
plt.grid(True)
plt.savefig('build/Braggbed.pdf')

plt.clf()
###### Emissionsspektrum #####

daten = np.loadtxt('Ecu.csv', delimiter=',', skiprows=1)

theta_2 = daten[:,0]
I_2 = daten[:,1]

talpha = 40.4
tbeta = 44.8

plt.plot(theta_2, I_2, 'x', label='Messwerte')
plt.axvline(x=talpha, color='green', linestyle='-', label=r'$K_\beta$')
plt.axvline(x=tbeta, color='red', linestyle='-', label=r'$K_\alpha$')
plt.xlabel(r'$2 \theta /°$')
plt.ylabel(r'$Imps /s$')
plt.legend()
plt.grid(True)
plt.savefig('build/ECu.pdf')

plt.clf()
##### Detailsspektrum #####

daten_2 = np.loadtxt('Detailspektrum.csv', delimiter=',', skiprows=1)

theta_3 = daten_2[:,0]  #Daten von winkel 34-37 rausgenommen zur besseren übersicht, sowie die Winkel von 49-50
I_3 = daten_2[:,1]

talpha = 40.4
tbeta = 45

plt.plot(theta_3, I_3, 'x', label='Messwerte')
plt.axvline(x=talpha, color='green', linestyle='-', label=r'$K_\beta$')
plt.axvline(x=tbeta, color='red', linestyle='-', label=r'$K_\alpha$')
plt.xlabel(r'$2 \theta /°$')
plt.ylabel(r'$Imps /s$')
plt.legend()
plt.grid(True)
plt.savefig('build/Detailsp.pdf')

plt.clf()
###### min Wellenlänge und max E #####
tmin = ufloat(4.0, 0.1)
ta = ufloat(22.5,0.1)
tb = ufloat(20.2,0.1)
h = 6.626e-34
c = 2.997e8
d = 201.4e-12
e_0 = 1.602e-19

def Bragg(t):
    return 2*d*unp.sin(t)

#print('lamda für Kalpha: ',Bragg(ta*np.pi/180))
#print('lamda für Kbeta: ',Bragg(tb*np.pi/180))

def E(t):
    return h*c/(2*d*unp.sin(t*np.pi/180)*e_0)
    

#print('E_kalpha = ',E(ta) )
#print('E_kbeta = ',E(tb) )
ta_1 = ufloat(22.25,0.1)
ta_2 = ufloat(22.7,0.1)

tb_1 = ufloat(20,0.1)
tb_2 = ufloat(20.4,0.1)

#print('deltaE_kalpha = ',E(ta_2)-E(ta_1) )
#print('deltaE_kbeta = ',E(tb_2)-E(tb_1))
#
#print('A_Ka = ',E(ta)/abs((E(ta_2)-E(ta_1))))
#print('A_Kb = ',E(tb)/abs((E(tb_2)-E(tb_1))) )

E_abs = 8980.476
R = 13.6

s1 = 29 - unp.sqrt(E_abs/R)
#print("s1: " , s1)
s2 = 29 - 2* unp.sqrt((E_abs - E(ta))/R)
#print("s2: " , s2)
s3 = 29 - 3* unp.sqrt((E_abs - E(tb))/R)
#print("s3: " , s3)
###### Absorptionsspektrum ######
#### Brom Br ####

#print('----- Brom -------')
datenBr = np.loadtxt('AbsorberBr.csv', delimiter=',', skiprows=1)

theta_br = datenBr[:,0]
I_Br = datenBr[:,1]

I_maxBr = np.max(I_Br)
I_minBr = np.min(I_Br)
I_kBr = np.min(I_Br)+(np.max(I_Br)-np.min(I_Br))/2

#print('I_minBr = ',np.min(I_Br))
#print('I_maxBr = ',np.max(I_Br))
#print('I_kBr = ',I_kBr)

def sigmoid(x, a, b):
    return a*x+b 
x_br = [26.4, 26.6]
y_br = [20, 27]

from scipy.optimize import curve_fit
params, covariance_matrix = curve_fit(sigmoid, x_br, y_br)

uncertainties = np.sqrt(np.diag(covariance_matrix))

for name, value, uncertainty in zip('ab', params, uncertainties): 
    print(f'{name} = {value:8.3f} ± {uncertainty:.3f}')

t_br = (22.5 +904)/35
t_br1 = ufloat(26.47, 0.05)
#print('t_br =', t_br)
#print('E_kBr = ', E(t_br1/2))
E_kBr =  E(t_br1/2)

plt.plot(np.linspace(26.4,26.6,50),sigmoid(np.linspace(26.4, 26.6, 50), 35, -904), color ='green', label=r'Hilfsgerade')
plt.plot(t_br, I_kBr, 'rx', label=r'$I_K$')
plt.plot(theta_br, I_Br, 'bx', label='Messwerte')
plt.xlabel(r'$2 \theta /°$')
plt.ylabel(r'$Imps /s$')
plt.legend()
plt.grid(True)
plt.savefig('build/AbsorberBr.pdf')

plt.clf()

##### Gallium Ga #####
#print('--------- Gallium --------------')
datenGa = np.loadtxt('AbsorberGa.csv', delimiter=',', skiprows=1)

theta = datenGa[:,0]
I_ga = datenGa[:,1]

I_maxGa = np.max(I_ga)
I_minGa = np.min(I_ga)
I_kGa = np.min(I_ga)+(np.max(I_ga)-np.min(I_ga))/2

#print('I_minGa = ',np.min(I_ga))
#print('I_maxGa = ',np.max(I_ga))
#print('I_kGa = ',I_kGa)

x_ga = [34.4, 34.5]
y_ga = [50, 60 ]

from scipy.optimize import curve_fit
params, covariance_matrix = curve_fit(sigmoid, x_ga, y_ga)

uncertainties = np.sqrt(np.diag(covariance_matrix))

for name, value, uncertainty in zip('ab', params, uncertainties): 
    print(f'{name} = {value:8.3f} ± {uncertainty:.3f}')

t_ga = (54.5 + 3390)/100
t_ga1 = ufloat(t_ga, 0.05)
#print('t_ga =', t_ga)
#print('E_kGa = ', E(t_ga1/2))
E_kGa =  E(t_ga1/2)

plt.plot(np.linspace(34.4,34.5,50),sigmoid(np.linspace(34.4, 34.5, 50), 100, -3390), color ='green', label=r'Hilfsgerade')
plt.plot(t_ga, I_kGa, 'rx', label=r'$I_K$')
plt.plot(theta, I_ga, 'x', label='Messwerte')
plt.xlabel(r'$2 \theta /°$')
plt.ylabel(r'$Imps /s$')
plt.legend()
plt.grid(True)
plt.savefig('build/AbsorberGa.pdf')

plt.clf()

#### Zink Zn ######
#print('--------- Zink---------------')
datenZn = np.loadtxt('AbsorberZn.csv', delimiter=',', skiprows=1)

theta = datenZn[:,0]
I_Zn = datenZn[:,1]

I_maxZn = np.max(I_Zn)
I_minZn = np.min(I_Zn)
I_kZn = np.min(I_Zn)+(np.max(I_Zn)-np.min(I_Zn))/2

#print('I_minZn = ',np.min(I_Zn))
#print('I_maxZn = ',np.max(I_Zn))
#print('I_kZn = ',I_kZn)

x_zn = [40, 40.2]
y_zn = [358, 769 ]

from scipy.optimize import curve_fit
params, covariance_matrix = curve_fit(sigmoid, x_zn, y_zn)

uncertainties = np.sqrt(np.diag(covariance_matrix))

for name, value, uncertainty in zip('ab', params, uncertainties): 
    print(f'{name} = {value:8.3f} ± {uncertainty:.3f}')

t_zn = (471 + 81842)/2055
t_zn1 = ufloat(t_zn, 0.05)
#print('t_zn =', t_zn)
#print('E_kZn = ', E(t_zn1/2))
E_kZn =  E(t_zn1/2)

plt.plot(np.linspace(40,40.2,50),sigmoid(np.linspace(40, 40.2, 50), 2055, -81842), color ='green', label=r'Hilfsgerade')
plt.plot(t_zn, I_kZn, 'rx', label=r'$I_K$')
plt.plot(theta, I_Zn, 'x', label='Messwerte')
plt.xlabel(r'$2 \theta /°$')
plt.ylabel(r'$Imps /s$')
plt.legend()
plt.grid(True)
plt.savefig('build/AbsorberZn.pdf')

plt.clf()

#### Strontium Sr #####
#print('----. Strontium ----------')
datenSr = np.loadtxt('AbsorberSr.csv', delimiter=',', skiprows=1)

theta = datenSr[:,0]
I_Sr = datenSr[:,1]

I_maxSr = np.max(I_Sr)
I_minSr = np.min(I_Sr)
I_kSr = np.min(I_Sr)+(np.max(I_Sr)-np.min(I_Sr))/2

#print('I_minSr = ',np.min(I_Sr))
#print('I_maxSr = ',np.max(I_Sr))
#print('I_kSr = ',I_kSr)

x_sr = [22, 22.2]
y_sr = [65, 83 ]

from scipy.optimize import curve_fit
params, covariance_matrix = curve_fit(sigmoid, x_sr, y_sr)

uncertainties = np.sqrt(np.diag(covariance_matrix))

for name, value, uncertainty in zip('ab', params, uncertainties): 
    print(f'{name} = {value:8.3f} ± {uncertainty:.3f}')

t_sr = (72 + 1915)/90
t_sr1 = ufloat(t_sr, 0.05)
#print('t_sr =', t_sr)
#print('E_kSr = ', E(t_sr1/2))
E_kSr =  E(t_sr1/2)

plt.plot(np.linspace(22,22.2,50),sigmoid(np.linspace(22, 22.2, 50), 90, -1915), color ='green', label=r'Hilfsgerade')
plt.plot(t_sr, I_kSr, 'rx', label=r'$I_K$')
plt.plot(theta, I_Sr, 'x', label='Messwerte')
plt.xlabel(r'$2 \theta /°$')
plt.ylabel(r'$Imps /s$')
plt.legend()
plt.grid(True)
plt.savefig('build/AbsorberSr.pdf')

plt.clf()

alpha = 1/137
#print('------ Abschirmkonstante----------')

s_kBr = (E(t_br1/2))/(R) - (alpha**2 * 35**4)/(4)
s_kGa =(E(t_ga1/2))/(R) - (alpha**2 * 31**4)/(4)
s_kZn =(E(t_zn1/2))/(R) - (alpha**2 * 30**4)/(4)
s_kSr =(E(t_sr1/2))/(R) - (alpha**2 * 38**4)/(4)

#print('s_kBr', 35-unp.sqrt(s_kBr))
#print('s_kGa', 31-unp.sqrt(s_kGa))
#print('s_kZn', 30-unp.sqrt(s_kZn))
#print('s_kSr', 38-unp.sqrt(s_kSr))

Z = [30, 31, 35, 38 ]
E = [8986, 10396, 13442, 16070]

from scipy.optimize import curve_fit
params, covariance_matrix = curve_fit(sigmoid, np.sqrt(E), Z)

uncertainties = np.sqrt(np.diag(covariance_matrix))

for name, value, uncertainty in zip('ab', params, uncertainties): 
    print(f'{name} = {value:8.3f} ± {uncertainty:.3f}')
# a = 0.258 +-0,018  ; b = 5.202+- 1.965

asf = np.linspace(90, 140, 50)

plt.plot (np.sqrt(E), Z, 'rx', label=r'$Daten$')
plt.plot(asf, sigmoid(asf, 0.258, 5.202), label=r'Ausgleichsgerade')
plt.grid(True)
plt.xlabel(r'$\sqrt{E_{K}}$')
plt.ylabel(r'Ordnungszahl Z')
plt.savefig('build/Moseley.pdf')

d = ufloat(0.257578,0.018)
R = 1/d**2
print('R= ', R)
