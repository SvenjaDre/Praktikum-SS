import numpy as np
import scipy.constants as const 

def Lande(S,L,J):
    return (3*J*(J+1)+(S*(S+1)-L*(L+1)))/(2*J*(J+1))

def N(dichte,Mmol):
    return 2*const.N_A*dichte/Mmol
def chi(gj,N,J):
    return const.mu_0*(const.physical_constants["Bohr magneton"][0]*gj)**2*N*J*(J+1)/(3*const.k*293.15)

Quantenz=np.array([[1.5,6,4.5],[3.5,0,3.5],[2.5,5,7.5]])
dichte=np.array([[7.24],[7.4],[7.8]])
Mmol=np.array([[336.48],[362.49],[373.0]])

gj=Lande(Quantenz[:,0],Quantenz[:,1],Quantenz[:,2])
n=N(dichte[:],Mmol[:])*1e6

Tgj=np.reshape(gj,(3,1))
print(gj)
print(np.ravel(n))
ch=chi(gj,np.ravel(n),Quantenz[:,2])
print(ch)