import matplotlib.pyplot as plt
import numpy as np
from uncertainties import ufloat

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

plt.plot(theta_2, I_2, 'x', label='Messwerte')
plt.xlabel(r'$2 \theta /°$')
plt.ylabel(r'$Imps /s$')
plt.legend()
plt.grid(True)
plt.savefig('build/ECu.pdf')

plt.clf()
##### Detailsspektrum #####

daten_2 = np.loadtxt('Detailspektrum.csv', delimiter=',', skiprows=1)

theta_3 = daten_2[:,0]
I_3 = daten_2[:,1]
#Daten von winkel 34-37 rausgenommen zur besseren übersicht, sowie die Winkel von 49-50
plt.plot(theta_3, I_3, 'x', label='Messwerte')
plt.xlabel(r'$2 \theta /°$')
plt.ylabel(r'$Imps /s$')
plt.legend()
plt.grid(True)
plt.savefig('build/Detailsp.pdf')

plt.clf()

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