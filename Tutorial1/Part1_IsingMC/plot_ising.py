########## ICTP-SAIFR Minicourse on Machine Learning for Many-Body Physics ##########
### Roger Melko, Juan Carrasquilla and Lauren Hayward Sierens
### Tutorial 1: Print observables as a function of temperature for the Ising model
#####################################################################################

import matplotlib.pyplot as plt
import numpy as np

### Input parameters (these should be the same as in ising_mc.py): ###
T_list = np.linspace(5.0,0.5,19) #temperature list
L = 4                            #linear size of the lattice
N_spins = L**2                   #total number of spins
J = 1                            #coupling parameter

### Critical temperature: ###
Tc = 2.0/np.log(1.0 + np.sqrt(2))*J

### Observables to plot as a function of temperature: ###
energy   = []
mag      = []
specHeat = []
susc     = []

### Loop to read in data for each temperature: ###
for T in T_list:
  file = open('ising2d_L%d_T%.4f.txt' %(L,T), 'r')
  data = np.loadtxt( file )

  E   = data[:,1]
  M   = abs(data[:,2])

  energy.append  ( np.mean(E)/(1.0*N_spins) )
  mag.append     ( np.mean(M)/(1.0*N_spins) )
  
  # *********************************************************************** #
  # *** FILL IN THE ESTIMATORS FOR THE SPECIFIC HEAT AND SUSCEPTIBILITY *** #
  # *********************************************************************** #
  specHeat.append( 0 )
  susc.append    ( 0 )
#end loop over T

plt.figure(figsize=(8,6))

plt.subplot(221)
plt.axvline(x=Tc, color='k', linestyle='--')
plt.plot(T_list, energy, 'o-')
plt.xlabel('$T$')
plt.ylabel('$<E>/N$')

plt.subplot(222)
plt.axvline(x=Tc, color='k', linestyle='--')
plt.plot(T_list, mag, 'o-')
plt.xlabel('$T$')
plt.ylabel('$<M>/N$')

plt.subplot(223)
plt.axvline(x=Tc, color='k', linestyle='--')
plt.plot(T_list, specHeat, 'o-')
plt.xlabel('$T$')
plt.ylabel('$C/N$')

plt.subplot(224)
plt.axvline(x=Tc, color='k', linestyle='--')
plt.plot(T_list, susc, 'o-')
plt.xlabel('T')
plt.ylabel('$\chi/N$')

plt.suptitle('%d x %d Ising model' %(L,L))

plt.show()
