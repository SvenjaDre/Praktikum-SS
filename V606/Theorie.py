import numpy as np
import scipy.constants as const 

def Lande(S,L,J):
    return (3*J*(J+1)+(S*(S+1)-L*(L+1)))/(2*J*(J+1))

def N(dichte,Mmol):
    return const.N_A*dichte/Mmol
def chi(gj,N,J):
    return const.mu_0*(const.mu_B*gj)**2*N*J(J+1)/(3*const.k*293.15)

Quantenz=np.array([[1.5,6,4.5],[3.5,0,3.5],[2.5,5,7.5]])

gj=Lande(Quantenz[:,0],Quantenz[:,1],Quantenz[:,2])

print(np.reshape(gj,(3,1)))