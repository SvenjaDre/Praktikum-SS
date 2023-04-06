import pandas as pd
import matplotlib.pyplot as plt
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
x0=2.3e-2
E,x,params1,Fehler1=Auswertung(np_array1,x0)

plt.plot(np_array1[:,0], E,"rx" ,label='Energiemaxima')
x_p=np.linspace(0,1020)
plt.plot(x_p,x_p*params1[0]+params1[1] ,"b-" ,label=f'Lineare Regression:E={params1[0]:.3}'r"$\unit{\mega\electronvolt\per\milli\bar} \cdot$" f'p+{params1[1]:.3}'r"$\unit{\mega\electronvolt}$")
plt.xlabel(r"p/$\unit{\milli\bar}$")
plt.ylabel(r"E/$\unit{\mega\electronvolt}$")
plt.legend(loc='best')

plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('build/Energiemaxima1.pdf')
plt.clf()

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

data2 = pd.read_excel('V701.xlsx',sheet_name="Messung 2;3,2cm")
np_array2 = np.array(data2)
x0=3.2e-2
E,x,params1,Fehler1=Auswertung(np_array2,x0)

plt.plot(np_array2[:,0], E,"rx" ,label='Energiemaxima')
x_p=np.linspace(0,1020)
plt.plot(x_p,x_p*params1[0]+params1[1] ,"b-" ,label='Lineare Regression')
plt.xlabel("x")
plt.ylabel("y")
plt.legend(loc='best')

plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('build/Energiemaxima2.pdf')
plt.clf()

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
