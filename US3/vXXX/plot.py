import matplotlib.pyplot as plt
import numpy as np
from uncertainties import ufloat
import pandas as pd
import scipy.constants as const

def Data(Name):
    Daten1 = pd.read_csv(Name, skiprows=0, sep=";")
    Daten2 = Daten1.replace(",", ".", regex=True)
    return Daten2.to_numpy(dtype=np.float64)

def max_exponent_2d(array):
    max_exp = []
    for i in range(len(array[0])):
        max_exp.append(int(np.floor(np.log10(np.abs(array[:,i].max())))))
    return max_exp

def min_exponent_2d(array):
    min_exp = []
    for i in range(len(array[0])):
        non_zero_values = array[:,i][np.nonzero(array[:,i])]
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
                if (isinstance(cell, int) or (isinstance(cell, float) and cell.is_integer())) and exponent[i] <=5 :
                    formatted_row.append("{:.0f}".format(cell))
                elif exponent[i] < -2:
                    formatted_row.append("{:.2f}".format(cell * 10**-minexponent[i], minexponent[i]).replace(".", ","))
                elif exponent[i] >5:
                    formatted_row.append("{:.2f}".format(cell * 10**-minexponent[i], minexponent[i]).replace(".", ","))
                else:
                    formatted_row.append("{:.2f}".format(cell).replace(".", ","))
                i=i+1
            f.write(" & ".join(formatted_row))
            f.write(" \\\\\n")
    return minexponent
def array_to_latex_table_1D(array, filename):
    exponent=max_exponent_2d(array)
    minexponent=min_exponent_2d(array)
    with open(filename, "w") as f:
        formatted_array = []
        i=0
        for cell in array:
            formatted_array=[]
            if (isinstance(cell, int) or (isinstance(cell, float) and cell.is_integer())) and exponent[i] <= 5:
                formatted_array.append("{:.0f}".format(cell))
            elif exponent[i] < -2:
                formatted_array.append("{:.2f}".format(cell * 10**-minexponent[i], -minexponent[i]).replace(".", ","))
            elif exponent[i] >= 5:
                formatted_array.append("{:.2f}".format(cell * 10**-minexponent[i], -minexponent[i]).replace(".", ","))
            else:
                formatted_array.append("{:.2f}".format(cell).replace(".", ","))
           
            f.write(", ".join(formatted_array))
            f.write(" \\\\\n")
    return minexponent

def Plot1 (x, y, xlabel="", ylabel="", filepath=""):
    fig, ax =plt.subplots()
    ax.plot(x,y,"r.",markersize=8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.grid()
    plt.savefig(filepath)
    plt.clf()

def Plot2 (x, y1,y2, xlabel="", ylabel="", filepath="",label1="",label2=""):
    fig, ax =plt.subplots()
    ax.plot(x,y1,"r.",markersize=8,label=label1)
    ax.plot(x,y2,"b.",markersize=8,label=label2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.legend()
    plt.grid()
    plt.savefig(filepath)
    plt.clf()


def Dwinkel(The):
    cl=1800
    cp=2700
    return 90 - (180/np.pi)*np.arcsin(np.sin(The*(np.pi / 180))*cl/cp)

def v(df, f0,alp,cL):
    return df*cL/(2*f0*np.cos(alp*(np.pi / 180)))

def v0(R,vFl):
    return vFl/(np.pi*R**2)

def s(t):
    dt=t-12
    return (dt/4)*6e-3+30e-3

The=np.array([15,30,60])
f0=2e6
cL=1800
Ri=1e-2
alp=Dwinkel(The)
frunt=Data("content/US3v.csv")
exp1=array_to_latex_table(frunt, "content/Tabelle1.tex")
Si70=Data("content/US3SI70.csv")
Si45=Data("content/US3SI45.csv")
exp2=array_to_latex_table(np.append(Si45,Si70[:,1:],axis=1), "content/Tabelle3.tex")
print(exp1)
print(exp2)
v15=v(frunt[2,1:],f0,alp[0],cL)
v30=v(frunt[1,1:],f0,alp[1],cL)
v60=v(frunt[3,1:],f0,alp[2],cL)
Plot1(v15,frunt[2,1:]/np.cos((np.pi / 180)*alp[0]),r"Strömungsgeschwindigkeit v [m/s]",r"Dopplerverschiebung $\Delta \nu/\cos{\alpha}$ [Hz]","Geschwindigkeit15.pdf")
Plot1(v30,frunt[1,1:]/np.cos((np.pi / 180)*alp[1]),r"Strömungsgeschwindigkeit v [m/s]",r"Dopplerverschiebung $\Delta \nu/\cos{\alpha}$ [Hz]","Geschwindigkeit30.pdf")
Plot1(v60,frunt[3,1:]/np.cos((np.pi / 180)*alp[2]),r"Strömungsgeschwindigkeit v [m/s]",r"Dopplerverschiebung $\Delta \nu/\cos{\alpha}$ [Hz]","Geschwindigkeit60.pdf")

Tiefe=s(Si45[:,0])

Plot2(Tiefe*1000,Si45[:,1],Si70[:,1], "Messtiefe s [mm]",r"Streuintensität [1000 v$^2$/s]","Streui.pdf", "45% Pumpleistung","70% Pumpleistung")
Plot2(Tiefe*1000,Si45[:,2],Si70[:,2], "Messtiefe s [mm]",r"Strömungsgeschwindigkeit [m/s]","Stroev.pdf", "45% Pumpleistung","70% Pumpleistung")