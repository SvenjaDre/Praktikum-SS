import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from scipy.stats import norm
from scipy.stats import poisson
import numpy as np

def Auswertung(array,x0):
    x=x0*array[:,0]/1013
    E=4*array[:,1]/array[0,1]
    params1, covariance_matrix = np.polyfit(array[:,0], E, deg=1, cov=True)
    errors1 = np.sqrt(np.diag(covariance_matrix))
    for name, value, error in zip('ab', params1, errors1):
        print(f'{name} = {value:.3f} ± {error:.3f}')
    return E,x,params1,errors1

def linear(x,y):
    params1, covariance_matrix = np.polyfit(x, y, deg=1, cov=True)
    errors1 = np.sqrt(np.diag(covariance_matrix))
    for name, value, error in zip('ab', params1, errors1):
        print(f'{name} = {value:.3f} ± {error:.3f}')
    return params1,errors1



data1 = pd.read_excel('V701.xlsx',sheet_name="Messung 1;2,3cm")
np_array1 = np.array(data1)
np_array1[-1,0]=1013
x0=2.3e-2
E,x,params1,Fehler1=Auswertung(np_array1,x0)


fig,ax=plt.subplots(layout="constrained")

ax.plot(np_array1[:,0], E,"rx" ,label='Energiemaxima')
x_p=np.linspace(0,1020)
ax.plot(x_p,x_p*params1[0]+params1[1] ,"b-" ,label=f'Lineare Regression:E={params1[0]:.3}'r"$\unit{\mega\electronvolt\per\milli\bar} \cdot$" f'p+{params1[1]:.3}'r"$\unit{\mega\electronvolt}$")
ax.set_xlabel(r"p/$\unit{\milli\bar}$")
ax.set_ylabel(r"E/$\unit{\mega\electronvolt}$")
ax.legend(loc='best')
print(f"Der Energieverlust pro Meter beträgt {params1[0]*1013/(x0*100):.3}"r"$\unit{\mega\electronvolt\per\centi\meter}")

def ptox(p):
    return 2.3e-2*p/1013

def xtop(x):
    return x*1013/2.3e-2

secax = ax.secondary_xaxis(location='top', functions=(ptox, xtop))
secax.set_xlabel('Effektive Länge/m')

#ax.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
fig.savefig('build/Energiemaxima1.pdf')
fig.clf()

Z=np_array1[:,2]
plt.plot(x, Z,"rx" ,label='Zählrate')
x_p=np.linspace(0,3e-2)
y_p=np.ones(x_p.size)
params2,errors2=linear(x[-4:],Z[-4:])
y1=x_p*params2[0]+params2[1]
plt.plot(x_p,y1 ,"b-" ,label=f'Lineare Regression:'r"$\frac{\text{Detektionen}}{\text{2 Minuten}}$"r'$=\num{-9.13e6}$'r"$\unit{\per\meter} \cdot$" r"x$+\num{2.37e5}$")
y2=y_p*Z[0]/2
plt.plot(x_p,y2,"g-",label=f"Halbe Zählrate")
SP=(Z[0]/2-params2[1])/params2[0]
print(f"Schnittpunkt={SP:.3}")
Esp=SP/x0*1013*params1[0]+params1[1]
print(f"Energie bei mittlerer Reichweite={Esp:.3}")
plt.xlabel(r"x/$\unit{\meter}$")
plt.ylabel(r"$\frac{\text{Detektionen}}{\text{2 Minuten}}$")
plt.ylim(0,100000)
plt.xlim(0,2.5e-2)
plt.legend(loc='best')

plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('build/Zählrate1.pdf')
plt.clf()

print("\n")
print("Entfernung 2")
data2 = pd.read_excel('V701.xlsx',sheet_name="Messung 2;3,2cm")
np_array2 = np.array(data2)
x0=3.2e-2
E,x,params1,Fehler1=Auswertung(np_array2,x0)
print(x)
print(E)
fig,ax=plt.subplots(layout="constrained")

ax.plot(np_array2[:,0], E,"rx" ,label='Energiemaxima')
x_p=np.linspace(0,1020)
ax.plot(x_p,x_p*params1[0]+params1[1] ,"b-" ,label=f'Lineare Regression:E={params1[0]:.3}'r"$\unit{\mega\electronvolt\per\milli\bar} \cdot$" f'p+{params1[1]:.3}'r"$\unit{\mega\electronvolt}$")
ax.set_xlabel(r"p/$\unit{\milli\bar}$")
ax.set_ylabel(r"E/$\unit{\mega\electronvolt}$")
ax.legend(loc='best')
print(f"Der Energieverlust pro Meter beträgt {params1[0]*1013/(x0*100):.3}"r"$\unit{\mega\electronvolt\per\centi\meter}")

def ptox(p):
    return 3.2e-2*p/1013

def xtop(x):
    return x*1013/3.2e-2

secax = ax.secondary_xaxis(location='top', functions=(ptox, xtop))
secax.set_xlabel('Effektive Länge/m')

#ax.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
fig.savefig('build/Energiemaxima2.pdf')
fig.clf()

Z=np_array2[:,2]
plt.plot(x, Z,"rx" ,label='Zählrate')
x_p=np.linspace(0,3e-2)
y_p=np.ones(x_p.size)
params2,errors2=linear(x[-5:],Z[-5:])
y1=x_p*params2[0]+params2[1]
plt.plot(x_p,y1 ,"b-" ,label=f'Lineare Regression:'r"$\frac{\text{Detektionen}}{\text{2 Minuten}}$"r'$=\num{-9.13e6}$'r"$\unit{\per\meter} \cdot$" r"x$+\num{2.37e5}$")
y2=y_p*Z[0]/2
plt.plot(x_p,y2,"g-",label=f"Halbe Zählrate")
SP=(Z[0]/2-params2[1])/params2[0]
print(f"Schnittpunkt={SP:.3}")
Esp=SP/x0*1013*params1[0]+params1[1]
print(f"Energie bei mittlerer Reichweite={Esp:.3}")
plt.xlabel(r"x/$\unit{\meter}$")
plt.ylabel(r"$\frac{\text{Detektionen}}{\text{2 Minuten}}$")
plt.ylim(0,60000)
plt.xlim(0,3e-2)
plt.legend(loc='best')

plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('build/Zählrate2.pdf')
plt.clf()

print("Statistik des radioaktiven Zerfalls")
data3 = pd.read_excel('V701.xlsx',sheet_name="Messung 3")
np_array3 = np.array(data3)

Mittel=np.mean(np_array3)
variance = np.var(np_array3)
std=np.sqrt(variance)
print(f"Mittelwert={Mittel:.0f}")
print(f"Varianz={variance:.0f}")
x = np.linspace(Mittel - 3*std, Mittel + 3*std, 1000)
y = norm.pdf(x, Mittel, std)

fig,ax=plt.subplots(layout="constrained")
ax.hist(np_array3[:], 17,range=(3800,4700),label="Datensatz")
ax2 = ax.twinx()
ax2.plot(x,y,label="Normalverteilung",color="r")
ax2.set_ylabel(r"Wahrscheinlichkeit")

ax.set_xlabel(r"Anzahl an Detektionen in 10s")
ax.set_ylabel(r"Häufigkeit eines Ereignisses")

fig.legend(loc=(0.57,0.8))

fig.tight_layout()
fig.savefig('build/Statistik.pdf')
fig.clf()

Mittel=Mittel

k=np.arange(3809, 4709,50,dtype=int)
y = poisson.pmf(k, mu=Mittel)

fig,ax=plt.subplots(layout="constrained")
ax.hist(np_array3[:], 17,range=(3800,4700),label="Datensatz")
ax2 = ax.twinx()
ax2.plot(k,y,"ro",label="Poissonverteilung")
ax2.set_ylabel(r"Wahrscheinlichkeit")

ax.set_xlabel(r"Anzahl an Detektionen in 10s")
ax.set_ylabel(r"Häufigkeit eines Ereignisses")

fig.legend(loc=(0.6,0.8))

fig.tight_layout()
fig.savefig('build/Statistik1.pdf')
fig.clf()