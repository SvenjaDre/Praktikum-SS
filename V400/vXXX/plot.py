import matplotlib.pyplot as plt
import numpy as np

#-----Brechungsgesetz-----

n1 = 1.42
n2 = 1.46
n3 = 1.49
n4 = 1.49
n5 = 1.48
n6 = 1.47
n7 = 1.25

n = (n1+n2+n3+n4+n5+n6+n7)/7
#print('Brechungsindex: ',n )
c =  2.9979 * 1e8
v = c/n
nt = 1.49
vt = c/nt
#print('v Plexi: ',v)
#print('v Plexi Theorie: ',vt)
#print('abw Tv: ', (v-vt)/vt)    #3,677932405566601 %
#--------Planparallele Platten----------
d = 5.85  #in cm
a1 = 10
a2 = 20
a3 = 30
a4 = 40
a5 = 50
a6 = 60
a7 = 70

b1 = 7
b2 = 13.5
b3 = 19.5
b4 = 25.5
b5 = 31
b6 = 36
b7 = 48.5

s1 = d*np.sin(a1-b1)/np.cos(b1)
s2 = d*np.sin(a2-b2)/np.cos(b2)
s3 = d*np.sin(a3-b3)/np.cos(b3)
s4 = d*np.sin(a4-b4)/np.cos(b4)
s5 = d*np.sin(a5-b5)/np.cos(b5)
s6 = d*np.sin(a6-b6)/np.cos(b6)
s7 = d*np.sin(a7-b7)/np.cos(b7)

#print('s1 = ', s1)
#print('s2 = ', s2)
#print('s3 = ', s3)
#print('s4 = ', s4)
#print('s5 = ', s5)
#print('s6 = ', s6)
#print('s7 = ', s7)


g = (540.9 + 520.5 + 521.1 + 514.5 + 523.3 + 479.2 + 492.6 + 498.6 + 500.7) / 9
r =  (635.3 + 610.2 + 604.5 + 523.3 + 566.0 + 578.8 + 604.8 +716.7 )/9
print('lg = ', g)
print('lr = ', r)














