########## Machine Learning for Quantum Matter and Technology  ######################
### Juan Carrasquilla, Estelle Inack, Giacomo Torlai, Roger Melko
### with code by Estelle Inack  inspired from https://github.com/agdelma/qmc_ho
###
### Tutorial 5:  Variational Monte Carlo for the harmonic oscillator
######################################################################################
import numpy as np
import matplotlib.pyplot as plt

red,blue,green = '#e85c47','#4173b2','#7dcca4'
############# PARAMETERS DEFINITION #######################################
α = 0.1
num_walkers = 500
num_MC_steps = 100

num_sgd_steps =100   # number of times to perform SGD
num_equil_steps = 10

learning_rate = 0.1

########### DEFINITION OF THE LOCAL ENERGY ############################
def EL(x,α):
    return 0.5*α + 0.5*x**2*(1-α**2)

########### DEFINITION OF THE TRANSITION PROBABILITY ############################
def transition_probability(x,x̄,α):
    return np.exp(-α*(x̄**2-x**2))

########### DEFINITION OF THE FORCE ############################
def Force(x,α):
    return -0.5*x**2

########### DEFINITION OF THE EXACT DERIVATIVE OF THE VARIATIONAL ENERGY ############
def true_der_α(x,α): 
    pass

########### DEFINITION OF THE METROPOLIS ALGORITHM ############################
def metropolis(num_walkers,walk,α,δ=1.0):
    num_accepted = 0        
    
    avg_e  = 0.0
    avg_f  = 0.0
    avg_ef = 0.0
    
    # generate new walker positions with box move
    new_walkers = -walk + np.random.rand(num_walkers)
    
    # generate new walker positions with gaussian moves
    #new_walkers = np.random.normal(loc=walk, scale=δ, size=num_walkers)
    
    # generate new walker positions with lorentzian move
    #new_walkers = np.random.standard_cauchy(size=num_walkers)
        
   # test new walkers
    for i in range(num_walkers):
        if np.random.random() < transition_probability(walk[i],new_walkers[i],α):
            num_accepted += 1
            walk[i] = new_walkers[i]
        avg_e  += EL(walk[i],α)
        avg_f  += Force(walk[i],α)
        avg_ef += EL(walk[i],α)*Force(walk[i],α)
        
    # Stochatic estimate of the derivative of the variational energy    
    der = 2*(avg_ef/num_walkers - (avg_e*avg_f)/num_walkers**2)
    
    return walk, avg_e/num_walkers, der

########  VMC ALGORITHM  ######################################
def vmc(walk,num_walkers,num_MC_steps,num_relax_steps,α):
    
    eng   = [] 
    der_α = []
    
    for step in range(num_MC_steps):
        walk, energy, der = metropolis(num_walkers,walk,α)
        if step>= num_relax_steps and step%num_relax_steps==0:
            eng= np.append(eng,[energy])
            der_α = np.append(der_α,[der])    
    return walk, eng, der_α


##############   Stochatic Gradient Descent  ######################
def sgd(walk,num_walkers,num_sgd_steps,num_equil_steps,learning_rate,α):
    walk, energy, der_α  = vmc(walk,num_walkers,num_sgd_steps,num_equil_steps,α)
    
    der = np.average(der_α)
    #Update the learning rate
    α -= learning_rate*der
    
    return walk, α, np.average(energy) 


#############################################################################
################################## TRAINING ##################################
##############################################################################

# initialize walkers uniformily in space between [-0.5:0.5]
walkers = -0.5 + np.random.rand(num_walkers)

############ Function for plotting: ############
def updatePlot(walkers):

    ### Plot the density distribution of walkers: ###
    plt.subplot(121)
    x = np.linspace(-4,4,1000)
    plt.plot(x,np.exp(-x**2)/np.sqrt(np.pi), '-',linestyle='solid', linewidth=5, color=green, zorder=-10, 
         label=r'${|\phi_0(x)|}^2$')
    plt.hist(walkers, density='true',bins='auto', color=red, alpha=0.7)
    plt.legend(loc='best', frameon=False)
    plt.ylim(0, 1)
    plt.title("Density distribution of the walkers")
    plt.ylabel('Probability')
    plt.xlabel('x')

    ### Plot the variational energy during training: ###
    plt.subplot(222)
    plt.plot(sgd_list,var_eng_training,'o-')
    plt.xlabel('SGD steps')
    plt.ylabel('Variational energy')

    ### Plot the variational parameter:  ###
    plt.subplot(224)
    plt.plot(sgd_list,α_training,'o-')
    plt.xlabel('SGD steps')
    plt.ylabel('α')
############ End of plotting function ############      
    
### Train for several SGD steps: ###

sgd_list         = []
α_training       = []
var_eng_training = []
k = 10

for i in range(num_sgd_steps):
    # Run SGD
    walkers, α, var_eng = sgd(walkers,num_walkers,num_MC_steps,num_equil_steps,learning_rate,α)

    ### Update the plot and print results every k SGD steps: ###
    if i % k == 0 or i==1:
    
        print( "Iteration %d:\n  α %f\n  E_var(α) %f\n" % (i, α, var_eng) )
    
        sgd_list.append(i)
        α_training.append(α)
        var_eng_training.append(var_eng)
        
        ### Update the plot of the resulting classifier: ###
        fig = plt.figure(2,figsize=(10,5))
        fig.subplots_adjust(hspace=.3,wspace=.3)
        plt.clf()
        updatePlot(walkers)
        plt.pause(0.1)

plt.savefig('harmonic_oscillator_results.pdf') # Save the figure showing the results in the current directory

plt.show()
