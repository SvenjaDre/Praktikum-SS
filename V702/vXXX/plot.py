import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.constants as const
import uncertainties.unumpy as unp
from uncertainties import ufloat
import matplotlib.ticker as ticker
import math

def Data(Name):
    Daten1 = pd.read_csv(Name, skiprows=0, sep=";")
    Daten2 = Daten1.replace(",", ".", regex=True)
    return Daten2.to_numpy(dtype=np.float64)

def max_exponent_2d(array):
    max_exp = []
    for i in range(len(array[0])):
        max_exp.append(int(np.floor(np.log10(np.abs(array[~np.isnan(array[:,0]),i].max())))))
    return max_exp

def min_exponent_2d(array):
    min_exp = []
    for i in range(len(array[0])):
        non_zero_values = array[~np.isnan(array[:,0]),i][np.nonzero(array[~np.isnan(array[:,0]),i])]
        if len(non_zero_values) == 0:
            min_exp.append(0)
        else:
            min_exp.append(int(np.floor(np.log10(np.abs(abs(non_zero_values).min())))))
    return min_exp

def array_to_latex_table(array, filename):
    exponent=max_exponent_2d(array)
    minexponent=min_exponent_2d(array)
    with open(filename, "w") as f:
        for row in array:
            formatted_row = []
            i=0
            for cell in row:
                if np.isnan(cell):
                    formatted_row.append("")
                else:
                    if (isinstance(cell, int) or (isinstance(cell, float) and cell.is_integer())) and exponent[i] <=5 :
                        formatted_row.append("{:.0f}".format(cell))
                    elif exponent[i] < -2:
                        formatted_row.append("{:.2f}".format(cell * 10**-minexponent[i], minexponent[i]).replace(".", ","))
                    elif exponent[i] >5:
                        formatted_row.append("{:.2f}".format(cell * 10**-minexponent[i], minexponent[i]).replace(".", ","))
                    elif (10*cell).is_integer():
                        formatted_row.append("{:.1f}".format(cell).replace(".", ","))
                    else:
                        formatted_row.append("{:.2f}".format(cell).replace(".", ","))
                i=i+1
            f.write(" & ".join(formatted_row))
            f.write(" \\\\\n")
    return minexponent

def array_to_latex_table_3d(array, filename):
    exponent=max_exponent_2d(array[:,:,0].T)
    minexponent=min_exponent_2d(array[:,:,0].T)
    with open(filename, "w") as f:
        for row in zip(*array):
            formatted_row = []
            i=0
            for cell in row:
                if np.isnan(cell[0]):
                    formatted_row.append("")
                else:
                    if (isinstance(cell[0], int) or (isinstance(cell[0], float) and cell[0].is_integer())) and exponent[i] <= 5:
                        formatted_row.append("${:.0f} \\pm {:.0f}$".format(cell[0], cell[1]))
                    elif exponent[i] < -2:
                        formatted_row.append("${:.2f} \\pm {:.2f}$".format(cell[0] * 10**-minexponent[i], cell[1] * 10**-minexponent[i], minexponent[i]).replace(".", ","))
                    elif exponent[i] >= 5:
                        formatted_row.append("${:.2f} \\pm {:.2f}$".format(cell[0] * 10**-minexponent[i], cell[1] * 10**-minexponent[i], minexponent[i]).replace(".", ","))
                    elif (10*cell[0]).is_integer():
                            formatted_row.append("${:.1f}\\pm{:.1f}".format(cell).replace(".", ","))
                    else:
                        formatted_row.append("${:.2f} \\pm {:.2f}$".format(cell[0], cell[1]).replace(".", ","))
                i=i+1
            f.write(" & ".join(formatted_row))
            f.write(" \\\\\n")
    return minexponent

def Plot1 (high,x, y, xlabel="", y1label="",label="", filepath=""):
    fig, ax1 =plt.subplots()
    fig.tight_layout()
    fig.subplots_adjust(right=0.8)
    mask=(unp.nominal_values(y)-unp.std_devs(y))>0
    ax1.errorbar(x[mask],unp.nominal_values(y[mask]),yerr=unp.std_devs(y[mask]),xerr=None,color="r",fmt=".",label=label)
    ax1.errorbar(x[~mask],unp.nominal_values(y[~mask]),yerr=unp.std_devs(y[~mask]),xerr=None,color="k",fmt=".",label="Messpunkte ohne Aussagekraft")
    if high:ax1.errorbar(x[high-1],unp.nominal_values(y[high-1]),yerr=unp.std_devs(y[high-1]),xerr=None,color="g",fmt=".",label="Beginn der linearen Regression des langen Zerfalls")
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(y1label)

    ax1.grid()
    return fig ,ax1,mask

def Plot2 (fig,ax,x,y,label1="",xlim1=[],ylim=[],color=""):
    ax.plot(x,y,label=label1,color=color)
    ax.set_xlim(xlim1)
    ax.set_ylim(ylim)


def linear_regression(x,y,color):
    params, covariance_matrix = np.polyfit(x, unp.nominal_values(y),w=1/unp.std_devs(y), deg=1, cov=True)

    errors = np.sqrt(np.diag(covariance_matrix))

    for name, value, error in zip('ab', params, errors):
        print(f'{name}_{color} = {value:.3g} ± {error:.3g}')
    return params,errors

def linear_regression2(x,y,color):
    params, covariance_matrix = np.polyfit(x, unp.nominal_values(y), deg=1, cov=True)

    errors = np.sqrt(np.diag(covariance_matrix))

    for name, value, error in zip('ab', params, errors):
        print(f'{name}_{color} = {value:.3g} ± {error:.3g}')
    return params,errors

def ImportDetection(Daten, Aktiv):
    udata=unp.uarray(Daten, np.sqrt(Daten))
    return udata-Aktiv

def Halbwertzeit(lamb,error):
    t= np.log(2)/lamb
    dt= t*(error/lamb)
    return t,dt

def Auswertung(Daten,color):
    mask=Daten[:,1]>=0
    #fig,ax=Plot1(Daten[mask,0],np.sqrt(Daten[mask,1]),r"U/V",r"$\sqrt{I/\text{pA}}$","Datenpunkte",f"content/{color}.pdf")
    goodU=Daten[mask,0]
    goodI=Daten[mask,1]
    params,errors=linear_regression(goodU[:(high)],np.sqrt(goodI[:(high)]),color="")
    x=np.linspace(min(Daten[mask,0])-1,max(Daten[mask,0])+1,1000)
    #Plot2(fig,ax,x,x*params[0]+params[1],f"content/{color}.pdf","Ausgleichsgerade",[min(Daten[mask,0])-0.5,max(Daten[mask,0])+0.5],[min(np.sqrt(Daten[mask,1]))-0.5,max(np.sqrt(Daten[mask,1]))+0.5])
    return params,errors

def Nullmessung(daten,dt):
    s=np.sum(daten)
    t=len(daten)*dt
    ds=np.sqrt(s)
    Aktivität=s/t
    dA=Aktivität*ds/s
    return ufloat(Aktivität,dA)

def Vanadium(Daten,dt):
    x=np.arange(dt,(len(Daten)+1)*dt,dt)
    y=unp.log(Daten)
    fig,ax,mask=Plot1(0,x,y,r"t/s",r"$\ln{N}$","Datenpunkte",f"content/Vanadium.pdf")
    params,errors=linear_regression(x,y,"Vanadium")
    x1=np.linspace(-1,max(x)+1,500)
    Plot2(fig,ax,x1,x1*params[0]+params[1],"Ausgleichsgerade",[0,max(x)+5],[0,max(unp.nominal_values(y))+0.5],None)
    Time,dT=Halbwertzeit(params[0],errors[0])
    print(f"Halbwertzeit von Vanadium:{Time:.3g} ± {dT:.3g}")
    ax.legend()
    plt.savefig(f"content/Vanadium.pdf")
    plt.clf()    
def Silber(Daten,dt,file,cut,high):
        x=np.arange(dt,(len(Daten)+1)*dt,dt)
        y=unp.log(Daten)
        fig,ax,mask=Plot1(cut,x,y,r"t/s",r"$\log{N}$","Datenpunkte",file)
        mask1=mask[cut:]
        xx1=x[cut:]
        yy1=y[cut:]
        params1,errors1=linear_regression(xx1[mask1],yy1[mask1],"Silber Ag108")
        newy=y-x*params1[0]-params1[1]
        xx2=x[:high+1]
        yy2=newy[:high+1]
        params2,errors2=linear_regression(xx2,yy2,"Silber Ag110")
        x1=np.linspace(-1,max(x)+5,500)
        ax.errorbar(x[high-1],unp.nominal_values(y[high-1]),yerr=unp.std_devs(y[high-1]),xerr=None,color="purple",fmt=".",label="Tmax")
        Plot2(fig,ax,x1,x1*params1[0]+params1[1],"Ausgleichsgerade Ag108",[0,max(x)+0.5],[0,max(unp.nominal_values(y))+0.5],"skyblue")
        Plot2(fig,ax,x1,x1*params2[0]+params2[1]+params1[1],"Ausgleichsgerade Ag110",[0,max(x)+0.5],[0,max(unp.nominal_values(y))+0.5],"steelblue")
        Time08,dT08=Halbwertzeit(params1[0],errors1[0])
        print(f"Halbwertzeit von Ag 108:{Time08:.3g} ± {dT08:.3g}")
        Time10,dT10=Halbwertzeit(params2[0],errors2[0])
        print(f"Halbwertzeit von Ag 110:{Time10:.3g} ± {dT10:.3g}")
        ax.legend(fontsize="small")
        plt.savefig(file)
        plt.clf() 


Zerfälle=Data("content/V702.CSV")
exp=array_to_latex_table(Zerfälle,"content/Tabelle1.tex")
NullN=Zerfälle[~np.isnan(Zerfälle[:,0]),0]
VanadiumN=Zerfälle[~np.isnan(Zerfälle[:,1]),1]
Silber1N=Zerfälle[~np.isnan(Zerfälle[:,2]),2]
Silber2N=Zerfälle[~np.isnan(Zerfälle[:,3]),3]
tNull=30
tVan=35
tS1=10
tS2=8

Hintergrundaktivität=Nullmessung(NullN,tNull)
print(f"Hintergrundaktivität:{Hintergrundaktivität}")
VaN=unp.uarray(VanadiumN,np.sqrt(VanadiumN))-Hintergrundaktivität*tVan
Si1N=unp.uarray(Silber1N,np.sqrt(Silber1N))-Hintergrundaktivität*tS1
Si2N=unp.uarray(Silber2N,np.sqrt(Silber2N))-Hintergrundaktivität*tS2

nVaN=np.pad(VaN, (0, len(Si2N) - len(VaN)), mode='constant', constant_values=math.nan)
nSi1N=np.pad(Si1N, (0, len(Si2N) - len(Si1N)), mode='constant', constant_values=math.nan)
array=np.array([
    np.column_stack((unp.nominal_values(nVaN),unp.std_devs(nVaN))),
    np.column_stack((unp.nominal_values(nSi1N),unp.std_devs(nSi1N))),
    np.column_stack((unp.nominal_values(Si2N),unp.std_devs(Si2N)))
])
exp2=array_to_latex_table_3d(array,"content/Tabelle2.tex")


Vanadium(VaN,tVan)
Silber(Si1N,tS1,"content/Silber1.pdf",19,10)
Silber(Si2N,tS2,"content/Silber2.pdf",24,13)



