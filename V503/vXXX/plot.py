import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import scipy.constants as const
from scipy.optimize import minimize_scalar
import numpy as np
import math

def Data():
    Daten1 = pd.read_csv("content/V503.CSV", skiprows=0, sep=";")
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
def array_to_latex_table_3d(array, filename):
    exponent=max_exponent_2d(array[:,:,0].T)
    minexponent=min_exponent_2d(array[:,:,0].T)
    with open(filename, "w") as f:
        for row in zip(*array):
            formatted_row = []
            i=0
            for cell in row:
                if (isinstance(cell[0], int) or (isinstance(cell[0], float) and cell[0].is_integer())) and exponent[i] <= 5:
                    formatted_row.append("${:.0f} \\pm {:.0f}$".format(cell[0], cell[1]))
                elif exponent[i] < -2:
                    formatted_row.append("${:.2f} \\pm {:.2f}$".format(cell[0] * 10**-minexponent[i], cell[1] * 10**-minexponent[i], minexponent[i]).replace(".", ","))
                elif exponent[i] >= 5:
                    formatted_row.append("${:.2f} \\pm {:.2f}$".format(cell[0] * 10**-minexponent[i], cell[1] * 10**-minexponent[i], minexponent[i]).replace(".", ","))
                else:
                    formatted_row.append("${:.2f} \\pm {:.2f}$".format(cell[0], cell[1]).replace(".", ","))
                i=i+1
            f.write(" & ".join(formatted_row))
            f.write(" \\\\\n")
    return minexponent

def plot_data_with_errorbars(good,good2,x, y, x_err, y_err, xlabel="", ylabel="", title="", filepath=""):
    fig, ax = plt.subplots()
    ax.errorbar(x, y, xerr=x_err, yerr=y_err, fmt='ko',label="Bedingung nicht erfüllt")
    ax.errorbar(x[good], y[good], xerr=x_err[good], yerr=y_err[good], fmt='go',label="Bedingung erfüllt/Abweichung zu groß")
    ax.errorbar(x[good2], y[good2], xerr=x_err[good2], yerr=y_err[good2], fmt='ro',label="gute Tropfen")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.grid()
    plt.legend(fontsize="small")
    plt.savefig(filepath)
    plt.clf()

def gemeinsamer_faktor(daten):
    def abstand(x):
        return np.sum(np.abs(np.round(daten / x) - daten / x))

    # Suche die reelle Zahl, die den Abstand zwischen den Datenpunkten und ihren gerundeten ganzzahligen Vielfachen minimiert
    result = minimize_scalar(abstand, bounds=(0, np.min(daten)))
    gesch_est = result.x

    return gesch_est

def Datapoint(Daten):
    mean = np.mean(Daten)
    std = np.std(Daten, ddof=1) / np.sqrt(len(Daten))
    return mean, std


def vx(t, dt, s):
    v0 = s / t
    dv = v0 * dt / t
    return v0, dv


def compare(vab, vauf):
    mdiff = vab[:, 0] - vauf[:, 0]
    ddiff = np.sqrt(vab[:, 1] ** 2 + vauf[:, 1] ** 2)
    msumm=vab[:,0]+vauf[:,0]
    return msumm,mdiff, ddiff


def test(v0, vab, vauf):
    good1 = vab[:, 0] > vauf[:, 0]
    msumm,mdiff, ddiff = compare(vab, vauf)
    diff = 2 * v0[:, 0] - mdiff
    good2 = abs(diff)-ddiff < 0
    good5 = np.logical_and.reduce((good1, good2))
    return msumm,mdiff, ddiff, diff, good5


def Vis0(T):
    return (T - 16) * (1.881e-5 - 1.805e-5) / 16 + 1.805e-5


def r(vis, g, oel, mdiff, ddiff):
    r = (3 / 2) * np.sqrt(vis * mdiff / (g * oel))
    dr = 0.5 * r * ddiff / mdiff
    return r,dr


def q0(vis, d, g, oel, U, msumm,mdiff, ddiff):
    q0 = 9 * np.pi * np.sqrt(vis ** 3 * mdiff)*msumm * d / (2 * np.sqrt(g * oel) * U)
    dq = q0*ddiff*np.sqrt(1/(4*mdiff**2)+1/(msumm**2))
    return q0, dq


def q(q0, r, dq0, dr):
    B = 6.17e-5*np.ones(16)
    p = 760*np.ones(16)
    q = q0 * np.sqrt((1 + B / (p * r)) ** 3)
    dq = np.sqrt(
        (q * dq0 / q0) ** 2 + (3 * q0 * B * dr / (2 * p * r**2)) ** 2 * (1 + B / (p * r))
    )
    return q, dq

def sortout(Raw,prep):
    mNew_Data=np.empty(np.shape(prep))
    mprep=prep[:,0]
    dprep=prep[:,1]
    mask=abs(Raw-mprep[:,np.newaxis])<abs(np.sqrt(len(Raw[0,:]))*dprep[:,np.newaxis])
    New_Data=np.where(mask,Raw,-1)
    for i in range (0,New_Data.shape[0]):
        mask1=New_Data>0
        mNew_Data[i,:]=Datapoint(New_Data[i,mask1[i]])
    return mNew_Data,New_Data

Daten = Data()
s = 5e-4
tauf = Daten[:, 4:7]
tab = Daten[:, 7:]
mtauf = np.empty([16, 2])
mtab = np.empty([16, 2])
vauf = np.empty([16, 2])
vab = np.empty([16, 2])
v0 = np.empty([16, 2])
for i in range(0, 16):
    mtauf[i] = np.array([Datapoint(tauf[i, :])])
    mtab[i] = np.array(Datapoint(tab[i, :]))
    vauf[i] = vx(mtauf[i, 0], mtauf[i, 1], s)
    vab[i] = vx(mtab[i, 0], mtab[i, 1], s)
    v0[i] = vx(Daten[i, 3], 0, s)
msumm,mdiff, ddiff, diff, good = test(v0, vab, vauf)
TC = np.array(
    [
        21,
        21.5,
        21.5,
        21.5,
        21.5,
        21.5,
        21.5,
        21.5,
        21.5,
        21.5,
        21.5,
        21.5,
        21.5,
        21.5,
        21.5,
        22,
    ]
)
vis0 = Vis0(TC)
oel = 886*np.ones(16)
g = const.g*np.ones(16)
Radius=np.empty([16,2])
Radius[:,0],Radius[:,1]=r(vis0, g, oel, abs(mdiff), ddiff)
Ladung0=np.empty([16,2])
d=7.625e-3
u=Daten[:,2]
Ladung0[:,0],Ladung0[:,1]=q0(vis0, d, g, oel, u,msumm, abs(mdiff), ddiff)
Ladung=np.empty([16,2])
Ladung[:,0],Ladung[:,1]=q(Ladung0[:,0], Radius[:,0], Ladung0[:,1], Radius[:,1])


array1=np.column_stack((np.array(msumm),np.array(ddiff)))
array2=np.column_stack((np.array(mdiff),np.array(ddiff)))
Messwerte=np.array([mtauf,mtab,vauf,vab,v0])
Auswertung=np.array([array1,array2,Ladung,Radius])
exp1=array_to_latex_table(Daten, "content/Tabelle1.tex")
exp2=array_to_latex_table_3d(Messwerte, "content/Tabelle2.tex")
exp3=array_to_latex_table_3d(Auswertung, "content/Tabelle3.tex")

good1= Ladung[:,0]> Ladung[:,1]
good2=np.logical_and(good,good1)

plot_data_with_errorbars(good,good2,Radius[:,0], Ladung[:,0], Radius[:,1], Ladung[:,1], xlabel=r"Radius$/\mu m$", ylabel=r"Ladung$/10^{-18}C$", filepath="build/Ladungen.pdf")
print(exp1)
print(exp2)
print(exp3)
print(good2)

e=gemeinsamer_faktor(Ladung[good2,0])
print(e)
F=const.physical_constants["Faraday constant"][0]
print(f"Avogadrokonstante:{F/e}")



# Nmtauf,Ntauf=sortout(tauf,mtauf)
# Nmtab,Ntab=sortout(tab,mtab)

# Nvauf = np.empty([16, 2])
# Nvab = np.empty([16, 2])

# for i in range(0, 16):
#     Nvauf[i] = vx(Nmtauf[i, 0], Nmtauf[i, 1], s)
#     Nvab[i] = vx(Nmtab[i, 0], Nmtab[i, 1], s)
# Nmsumm,Nmdiff, Nddiff, Ndiff, Ngood = test(v0, Nvab, Nvauf)

# NRadius=np.empty([16,2])
# NRadius[:,0],NRadius[:,1]=r(vis0, g, oel, abs(Nmdiff), Nddiff)
# NLadung0=np.empty([16,2])
# NLadung0[:,0],NLadung0[:,1]=q0(vis0, d, g, oel, u,Nmsumm, abs(Nmdiff), Nddiff)
# NLadung=np.empty([16,2])
# NLadung[:,0],NLadung[:,1]=q(NLadung0[:,0], NRadius[:,0], NLadung0[:,1], NRadius[:,1])
# print(NLadung)
# Ne=gemeinsamer_faktor(NLadung[Ngood,0])
# print(Ne)