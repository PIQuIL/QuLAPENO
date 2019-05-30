########## Machine Learning for Quantum Matter and Technology  ######################
### Juan Carrasquilla, Estelle Inack, Giacomo Torlai, Roger Melko
### with code from Lauren Hayward Sierens and Juan Carrasquilla, with the lines performing
### PCA taken from Laurens van der Maaten's implementation of t-SNE in Python
###
### Tutorial 3 (solutions): This code performs principal component analysis (PCA) on spin configurations
### corresponding to the Ising model.
#####################################################################################



import numpy as np
import matplotlib.pyplot as plt

#Specify font sizes for plots:
plt.rcParams['axes.labelsize']  = 16
plt.rcParams['legend.fontsize'] = 16
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['font.size']       = 18

modelName = "Ising" 

#Parameters:
num_components = 2

### Loop over all lattice sizes: ###
for L in [20,40,80]: #Note: L=80 requires several minutes
    print("L=%d"%L)

    ### Read in the data from the files: ###
    X      = np.loadtxt("data_tutorial3/spinConfigs_%s_L%d.txt" %(modelName,L), dtype='int8')
    
    labels = np.loadtxt("data_tutorial3/temperatures_%s_L%d.txt" %(modelName,L), dtype='float')

    
    (N_configs, N_spins) = X.shape

    ### Perform the PCA: ###
    X_cent = X - np.tile(np.mean(X, 0), (N_configs, 1))
    (lamb, P) = np.linalg.eig(np.dot(X_cent.T, X_cent)/(N_configs-1.0))

    ### Sort according to decreasing order of the eigenvalues: ###
    indices_sorted = lamb.argsort()[::-1] #The [::-1] is to get the reverse order (largest eigenvalues first)
    lamb = lamb[indices_sorted]
    P = P[:,indices_sorted]

    ### Get the principal components (columns of the matrix X_prime): ###
    X_prime = np.dot(X_cent, P[:,0:num_components])
    
    ### PLOT FIGURE FOR PART C (explained variance ratios): ###
    plt.figure(1)
    ratios = lamb/np.sum(lamb)
    plt.semilogy(np.arange(N_spins), ratios, 'o-', label="L=%d"%L)

    ### PLOT FIGURE FOR PARTS A and B (first two principal components): ###
    plt.figure()
    plt.axes([0.17, 0.13, 0.81, 0.78]) #specify axes (for figure margins) in the format [xmin, ymin, xwidth, ywidth]
    #plt.scatter(X_prime[:, 0], X_prime[:, 1], s=40) #PART A
    sc = plt.scatter(X_prime[:, 0], X_prime[:, 1], c=labels, s=40, cmap=plt.cm.coolwarm) #PART B
    cb = plt.colorbar(sc, cmap=plt.cm.coolwarm) #PART B
    plt.title("L=%d"%L)
    plt.xlabel("x'_1")
    plt.ylabel("x'_2")
    plt.savefig("xPrime1_xPrime2_%s_L%d.pdf" %(modelName,L))

    ### PLOT FIGURE FOR PART D (elements of p1): ###
    plt.figure()
    plt.axes([0.19, 0.13, 0.79, 0.78]) #specify axes (for figure margins) in the format [xmin, ymin, xwidth, ywidth]
    plt.plot(np.arange(N_spins),np.abs(P[:,0]))
    plt.title("L=%d"%L)
    plt.xlabel("Component index")
    plt.ylabel("Absolute value of components of p1")
    plt.savefig("p1_%s_L%d.pdf"  %(modelName,L))

plt.figure(1)
plt.xlim([0,10])
plt.ylim([10**(-3),1])
plt.xlabel("Component index")
plt.ylabel("Explained variance ratio")
plt.legend()
plt.savefig("ratios_%s.pdf" %modelName)

plt.show()
