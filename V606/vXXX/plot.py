import matplotlib.pyplot as plt
import numpy as np

Daten = np.loadtxt('Messfilter.csv', delimiter=',', skiprows=1)

f = Daten[:,0]
U = Daten[:,1]

plt.plot(f, U, 'x')
plt.xlabel(r'$\nu /kHz$')
plt.ylabel(r'$U /mV$')
plt.grid(True)


plt.savefig('build/plot.pdf')
