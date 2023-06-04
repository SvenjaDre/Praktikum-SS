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

print('deltaE_kalpha = ',E(ta_2)-E(ta_1) )
print('deltaE_kbeta = ',E(tb_2)-E(tb_1))

print('A_Ka = ',E(ta)/abs((E(ta_2)-E(ta_1))))
print('A_Kb = ',E(tb)/abs((E(tb_2)-E(tb_1))) )

E_abs = 8980.476
R = 13.6

s1 = 29 - unp.sqrt(E_abs/R)
print("s1: " , s1)
s2 = 29 - 2* unp.sqrt((E_abs - E(ta))/R)
print("s2: " , s2)
s3 = 29 - 3* unp.sqrt((E_abs - E(tb))/R)
print("s3: " , s3)
###### Absorptionsspektrum ######
#### Brom Br ####

datenBr = np.loadtxt('AbsorberBr.csv', delimiter=',', skiprows=1)

theta = datenBr[:,0]
I = datenBr[:,1]

plt.plot(theta, I, 'x', label='Messwerte')
plt.xlabel(r'$2 \theta /°$')
plt.ylabel(r'$Imps /s$')
plt.legend()
plt.grid(True)
plt.savefig('build/AbsorberBr.pdf')

plt.clf()

##### Gallium Ga #####
datenGa = np.loadtxt('AbsorberGa.csv', delimiter=',', skiprows=1)

theta = datenGa[:,0]
I = datenGa[:,1]

plt.plot(theta, I, 'x', label='Messwerte')
plt.xlabel(r'$2 \theta /°$')
plt.ylabel(r'$Imps /s$')
plt.legend()
plt.grid(True)
plt.savefig('build/AbsorberGa.pdf')

plt.clf()

#### Zink Zn ######

datenZn = np.loadtxt('AbsorberZn.csv', delimiter=',', skiprows=1)

theta = datenZn[:,0]
I = datenZn[:,1]

plt.plot(theta, I, 'x', label='Messwerte')
plt.xlabel(r'$2 \theta /°$')
plt.ylabel(r'$Imps /s$')
plt.legend()
plt.grid(True)
plt.savefig('build/AbsorberZn.pdf')

plt.clf()

#### Strontium Sr #####

datenSr = np.loadtxt('AbsorberSr.csv', delimiter=',', skiprows=1)

theta = datenSr[:,0]
I = datenSr[:,1]

plt.plot(theta, I, 'x', label='Messwerte')
plt.xlabel(r'$2 \theta /°$')
plt.ylabel(r'$Imps /s$')
plt.legend()
plt.grid(True)
plt.savefig('build/AbsorberSr.pdf')

plt.clf()