######################################################################################
### Code taken from https://github.com/agdelma/qmc_ho
### modified by Estelle Inack 
###
###  Variational Monte Carlo for the harmonic oscillator
######################################################################################

import numpy as np
import matplotlib.pyplot as plt

red,blue,green = '#e85c47','#4173b2','#7dcca4'

def EL(x,α):
    return α + x**2*(0.5-2*α**2)

def transition_probability(x,x̄,α):
    return np.exp(-2*α*(x̄**2-x**2))

def vmc(num_walkers,num_MC_steps,num_equil_steps,α,δ=1.0):
    
    # initilaize walkers
    walkers = -0.5 + np.random.rand(num_walkers)
    
    # initialize energy and number of accepted updates
    estimator = {'E':np.zeros(num_MC_steps-num_equil_steps)}
    num_accepted = 0
    
    for step in range(num_MC_steps):
        
        # generate new walker positions 
        new_walkers = np.random.normal(loc=walkers, scale=δ, size=num_walkers)
        
        # test new walkers
        for i in range(num_walkers):
            if np.random.random() < transition_probability(walkers[i],new_walkers[i],α):
                num_accepted += 1
                walkers[i] = new_walkers[i]
                
            # measure energy
            if step >= num_equil_steps:
                measure = step-num_equil_steps
                estimator['E'][measure] = EL(walkers[i],α)
                
    # output the acceptance ratio
    print('accept: %4.2f' % (num_accepted/(num_MC_steps*num_walkers)))
    
    return estimator

α = 0.4
num_walkers = 400
num_MC_steps = 30000
num_equil_steps = 3000

np.random.seed(1173)

estimator = vmc(num_walkers,num_MC_steps,num_equil_steps,α)

#from scipy.stats import sem
Ē,ΔĒ = np.average(estimator['E']),np.std(estimator['E'])/np.sqrt(estimator['E'].size-1)

print('Ē = %f ± %f' % (Ē,ΔĒ))

Ēmin = []
ΔĒmin = []
α = np.array([0.45, 0.475, 0.5, 0.525, 0.55])
for cα in α: 
    estimator = vmc(num_walkers,num_MC_steps,num_equil_steps,cα)
    Ē,ΔĒ = np.average(estimator['E']),np.std(estimator['E'])/np.sqrt(estimator['E'].size-1)
    Ēmin.append(Ē)
    ΔĒmin.append(ΔĒ)
    print('%5.3f \t %7.5f ± %f' % (cα,Ē,ΔĒ))


cα = np.linspace(α[0],α[-1],1000)
plt.plot(cα,0.5*cα + 1/(8*cα), '-', linewidth=1, color=green, zorder=-10, 
         label=r'$\frac{\alpha}{2} + \frac{1}{8\alpha}$')
plt.errorbar(α,Ēmin,yerr=ΔĒmin, linestyle='None', marker='o', elinewidth=1.0, 
             markersize=6, markerfacecolor=blue, markeredgecolor=blue, ecolor=blue, label='VMC')
plt.xlabel(r'$\alpha$')
plt.ylabel('E')
plt.xlim(0.44,0.56)
plt.legend(loc='upper center')
plt.show() 
