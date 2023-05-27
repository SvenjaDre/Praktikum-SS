import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.constants as const
import uncertainties as unc
import uncertainties.unumpy as unp

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
                elif (10*cell).is_integer():
                    formatted_row.append("{:.1f}".format(cell).replace(".", ","))
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

def Plot1 (high,x, y, xlabel="", ylabel="",label="", filepath=""):
    fig, ax =plt.subplots()
    ax.plot(x,y,"r.",markersize=8,label=label)
    ax.plot(x[high-1],y[high-1],"g.",markersize=10,label="letzter Datenpunkt für lineare Regression")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.grid()
    return fig ,ax

def Plot2 (fig,ax,x,y, filepath="",label1="",xlim1=[],ylim=[]):
    ax.plot(x,y,label=label1)
    ax.set_xlim(xlim1)
    ax.set_ylim(ylim)
    plt.legend()
    plt.savefig(filepath)
    plt.clf()

def linear_regression(x,y,color):
    params, covariance_matrix = np.polyfit(x, y, deg=1, cov=True)

    errors = np.sqrt(np.diag(covariance_matrix))

    for name, value, error in zip('ab', params, errors):
        print(f'{name}_{color} = {value:.3g} ± {error:.3g}')
    return params,errors

def Auswertung(Daten,color,high):
    mask=Daten[:,1]>=0
    #fig,ax=Plot1(high,Daten[mask,0],np.sqrt(Daten[mask,1]),r"U/V",r"$\sqrt{I}$/pA","Datenpunkte",f"content/{color}.pdf")
    goodU=Daten[mask,0]
    goodI=Daten[mask,1]
    params,errors=linear_regression(goodU[:(high)],np.sqrt(goodI[:(high)]),color)
    x=np.linspace(min(Daten[mask,0])-1,max(Daten[mask,0])+1,1000)
    #Plot2(fig,ax,x,x*params[0]+params[1],f"content/{color}.pdf","Ausgleichsgerade",[min(Daten[mask,0])-0.5,max(Daten[mask,0])+0.5],[min(np.sqrt(Daten[mask,1]))-0.5,max(np.sqrt(Daten[mask,1]))+0.5])
    return params,errors

def Grenzspannung(params,color):
    UGr=-params[1]/params[0]
    print(f"Grenzspannung von {color} ist {UGr}")
    return UGr

def plot_data_with_errorbars(dict, xlabel="", ylabel=""):
    fig, ax = plt.subplots()
    for key in dict:
        if key == "Rot":
            ax.errorbar(const.c/dict[key][0], unp.nominal_values(dict[key][1]),unp.std_devs(dict[key][1]), fmt='ro',label=f"{key}({int(dict[key][0]*1e9)}nm)")
        else:
            ax.errorbar(const.c/dict[key][0], unp.nominal_values(dict[key][1]),unp.std_devs(dict[key][1]), fmt='ko',label=f"{key}({int(dict[key][0]*1e9)}nm)")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.grid()
    return fig,ax

def Photo(dict):
    fig,ax=plot_data_with_errorbars(dict,r"$\nu$/Hz","U/V")
    x=const.c/(np.array([dict["Gelb"][0],dict["Grün"][0],dict["Violett"][0]]))
    y=unp.nominal_values(np.array([dict["Gelb"][1],dict["Grün"][1],dict["Violett"][1]]))
    params, errors =linear_regression(x,y,"")
    x1=np.linspace(0,8e14,1000)
    Plot2 (fig,ax,x1,x1*params[0]+params[1], filepath="content/Grenzspannung.pdf",label1="Ausgleichsgerade",xlim1=[0,7e14],ylim=[-1.15,2])

def Plot3 (x, y, xlabel="", ylabel="", filepath=""):
    fig, ax =plt.subplots()
    ax.plot(x,y,"r.",markersize=6)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.grid()
    plt.savefig(filepath)

Daten2V=Data("content/2V.CSv")
green=Daten2V[:,:2]
lila=Daten2V[:,2:4]
red=Daten2V[:,4:]

green_mean, green_error = Auswertung(green, "grün", 7)
lila_mean, lila_error = Auswertung(lila, "lila", 12)
red_mean, red_error = Auswertung(red[14:,:], "rot", 3)

gp = unp.uarray(green_mean, green_error)
lp = unp.uarray(lila_mean, lila_error)
rp = unp.uarray(red_mean, red_error)


Ug=Grenzspannung(gp,"Grün")
Ul=Grenzspannung(lp,"Violett")
Ur=Grenzspannung(rp,"Rot")


exp1=array_to_latex_table(Daten2V,"content/Messwerte2V.tex")

Daten20V=Data("content/20V.CSV")

Daten20V=np.append(Daten20V,[[0,0]],axis=0)

gelb=Daten20V[54:86,:]

gelb_mean , gelb_error=Auswertung(gelb, "gelb", 7)

rgelb=unp.uarray(gelb_mean , gelb_error)

Ugelb=Grenzspannung(rgelb,"gelb")

D1,D2,D3,D4,D5=np.vsplit(np.array(Daten20V),5)
exp2=array_to_latex_table(np.column_stack([D1,D2,D3,D4,D5]),"content/Messwerte20V.tex")

wave={
    "Rot": [623e-9,Ur],
    "Gelb":[578e-9,Ugelb],
    "Grün":[546e-9,Ug],
    "Violett":[435e-9,Ul]
}

Photo(wave)

Plot3(Daten20V[:,0],Daten20V[:,1],"U/V","I/pA","content/Allgelb.pdf")

