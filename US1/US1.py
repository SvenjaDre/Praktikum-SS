import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from scipy.stats import norm
from scipy.stats import poisson
import numpy as np


def linear(x, y):
    params1, covariance_matrix = np.polyfit(x, y, deg=1, cov=True)
    errors1 = np.sqrt(np.diag(covariance_matrix))
    for name, value, error in zip("ab", params1, errors1):
        print(f"{name} = {value:.3f} ± {error:.3f}")
    return params1, errors1

def Data(Name):
    Daten1 = pd.read_csv(Name, skiprows=0, sep=";")
    Daten2 = Daten1.replace(",", ".", regex=True)
    return Daten2.to_numpy(dtype=np.float64)

daten=Data("Us1.CSV")
print(daten)

fig,ax= plt.subplots()

ax.plot(daten[:,1]/2,daten[:,0],"rx",label="Datenpunkte")

params,errors=linear(daten[:,1]/2,daten[:,0])
x=np.linspace(-5,30,200)
ax.plot(x,x*params[0]+params[1],label="Ausgleichsgerade")
ax.set_xlabel("$\mu$t/s")
ax.set_ylabel("s/mm")
ax.set_xlim([0,30])
ax.set_ylim([-5,80])

fig.savefig("Schallgeschwindigkeit.pdf")
fig.clf()

fig,ax= plt.subplots()

ax.plot(daten[:,3]/2,daten[:,2],"rx",label="Datenpunkte")

params1,errors1=linear(daten[:,3]/2,daten[:,2])
x=np.linspace(-5,30,200)
ax.plot(x,x*params1[0]+params1[1],label="Ausgleichsgerade")
ax.set_xlabel("$\mu$t/s")
ax.set_ylabel("s/mm")
ax.set_xlim([0,30])
ax.set_ylim([-5,80])

fig.savefig("Schallgeschwindigkeit2.pdf")
fig.clf()

C_A=(params[0]+params1[0])*1000/2
DC=np.std([params[0],params1[0]],ddof=1)/np.sqrt(2)
print(f"Schallgeschwindigkeit c={C_A} ± {DC}")
bm=(params[1]+params1[1])/2
print(f"Anpassungsschicht b={bm} ± {np.std([params[1],params1[1]],ddof=1)/np.sqrt(2)}")
c=2730

Mess_so=c*daten[:,1]*10**-6/2+bm*10**-3
Mess_su=c*daten[:,3]*10**-6/2+bm*10**-3
print(Mess_so*1000)

B=80e-3

d=(B-Mess_so-Mess_su)*1000

print(d)

print(f"relativer Fehler des Durchmessers:  {100*abs(d-daten[:,4])/daten[:,4]}")
print(f"relativer Fehler vom Ascan zur Tiefe:  {100*abs(Mess_so*1000-daten[:,0])/daten[:,0]}")
Bscan_T=np.array([
    63.6,
    55.8,
    47.7,
    40.1,
    31.2,
    22.5,
    14.5
]
)
print(f"relative Abweichung vom Bscan zur Schiebelehre{100*abs(Bscan_T-daten[:,2])/daten[:,2]}")