import numpy as np
import scipy.optimize
import uncertainties as unc
import uncertainties.unumpy as unp
from scipy import optimize
import matplotlib.pyplot as plt
from uncertainties import ufloat

Daten = np.loadtxt('Messfilter.csv', delimiter=',', skiprows=1)

f = Daten[:,0]
U = Daten[:,1]

plt.plot(f, U, 'x')
plt.xlabel(r'$\nu /kHz$')
plt.ylabel(r'$U /mV$')
plt.grid(True)


plt.savefig('build/plot.pdf')

n = 250 
F = 8.66*10**-5 #m^2 
I = 135 #mm
R_3 = 998 #ohm
U_sp = 1.5 # eingestellt
print('------------------Nd2O3--------------------')

pw_Nd2O3 = 7240  # kg/m^3
m_1 = 7.66*10**-3 #kg
L_1 = 0.18
Q1 = m_1/(L_1*pw_Nd2O3)
print('Q_real = ', Q1)
U_o1 = 5*10**-7
U_o2 = 1*10**-7
U_o3 = 1.5*10**-7
U_m1 = 10*10**-3
U_m2 = 12*10**-3
U_m3 = 19*10**-3
R_o1 = 659*5*10**-3
R_o2 = 661*5*10**-3
R_o3 = 668*5*10**-3
R_m1 = 646*5*10**-3
R_m2 = 644*5*10**-3
R_m3 = 646*5*10**-3

U_o=np.array([U_o1, U_o2, U_o3])
U_m=np.array([U_m1, U_m2, U_m3])
R_o=np.array([R_o1, R_o2, R_o3])
R_m=np.array([R_m1, R_m2, R_m3])

deltaU=U_m-U_o
deltaR=R_o-R_m
print('deltaU:', deltaU*1000)
print('deltaR:', deltaR*1000)

deltaUmittel = ufloat(np.mean(deltaU), np.std(deltaU))
deltaRmittel = ufloat(np.mean(deltaR), np.std(deltaR))

print('Umitel', deltaUmittel*1000)
print('Rmitel', deltaRmittel*1000)

X_U = (4*F*deltaUmittel)/(Q1*U_sp)*10**-2
print('sus X_u = ',X_U)
X_R=2*deltaRmittel/R_3*F/Q1
print('sus X_R = ',X_R)


print('---------------Dy2O3---------------')
m2 = 15.1*10**-3
p_wDy = 7800
L_2 = 0.175
Q2 = m2/(L_2*p_wDy)
print('Q_real = ', Q2)
U_o1 = 2*10**-7
U_o2 = 4*10**-7
U_o3 = 3*10**-7
U_m1 = 270*10**-3
U_m2 = 270*10**-3
U_m3 = 270*10**-3
R_o1 = 667*5*10**-3
R_o2 = 664*5*10**-3
R_o3 = 665*5*10**-3
R_m1 = 366*5*10**-3
R_m2 = 360*5*10**-3
R_m3 = 370*5*10**-3

U_o=np.array([U_o1, U_o2, U_o3])
U_m=np.array([U_m1, U_m2, U_m3])
R_o=np.array([R_o1, R_o2, R_o3])
R_m=np.array([R_m1, R_m2, R_m3])

deltaU=U_m-U_o
deltaR=R_o-R_m
print('deltaU:', deltaU*1000)
print('deltaR:', deltaR*1000)

deltaUmittel = ufloat(np.mean(deltaU), np.std(deltaU))
deltaRmittel = ufloat(np.mean(deltaR), np.std(deltaR))

print('Umitel', deltaUmittel*1000)
print('Rmitel', deltaRmittel*1000)

X_U = (4*F*deltaUmittel)/(Q2*U_sp)*10**-2
print('sus X_u = ',X_U)
X_R=2*deltaRmittel/R_3*F/Q2
print('sus X_R = ',X_R)


print('---------Gd2o3-----------')

m3 = 10.2*10**-3
p_wGd = 7400
L_3 = 0.175
Q3 = m3/(L_3*p_wGd)
print('Q_real = ', Q3)

U_o1 = 3*10**-7
U_o2 = 3*10**-7
U_o3 = 3*10**-7
U_m1 = 60*10**-3
U_m2 = 60*10**-3
U_m3 = 65*10**-3
R_o1 = 663*5*10**-3
R_o2 = 666*5*10**-3
R_o3 = 669*5*10**-3
R_m1 = 541*5*10**-3
R_m2 = 540*5*10**-3
R_m3 = 540*5*10**-3

U_o=np.array([U_o1, U_o2, U_o3])
U_m=np.array([U_m1, U_m2, U_m3])
R_o=np.array([R_o1, R_o2, R_o3])
R_m=np.array([R_m1, R_m2, R_m3])

deltaU=U_m-U_o
deltaR=R_o-R_m
print('deltaU:', deltaU*1000)
print('deltaR:', deltaR*1000)

deltaUmittel = ufloat(np.mean(deltaU), np.std(deltaU))
deltaRmittel = ufloat(np.mean(deltaR), np.std(deltaR))

print('Umitel', deltaUmittel*1000)
print('Rmitel', deltaRmittel*1000)

X_U = (4*F*deltaUmittel)/(Q3*U_sp)*10**-2
print('sus X_u = ',X_U)
X_R=2*deltaRmittel/R_3*F/Q3
print('sus X_R = ',X_R)


