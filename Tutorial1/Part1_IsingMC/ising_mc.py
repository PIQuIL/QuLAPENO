########## Machine Learning for Quantum Matter and Technology  ######################
### Juan Carrasquilla, Estelle Inack, Giacomo Torlai, Roger Melko 
### with code from Lauren Hayward Sierens
### Tutorial 1: Monte Carlo for the Ising model
#####################################################################################

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import random

### Input parameters: ###
T_list = np.linspace(5.0,2.0,7) #temperature list
T_list = [2.0]
L = 20                            #linear size of the lattice
N_spins = L**2                   #total number of spins
J = 1                            #coupling parameter

### Critical temperature: ###
Tc = 2.0/np.log(1.0 + np.sqrt(2))*J

### Monte Carlo parameters: ###
n_eqSweeps = 0   #number of equilibration sweeps
n_bins = 1000       #total number of measurement bins
n_sweepsPerBin=1  #number of sweeps performed in one bin

### Files to write training and testing spin configurations (X) and phases (y): ###
train_frac   = 2.0/7.0 #fraction of data to be used for training
file_Xtrain = open('Xtrain.txt', 'w')
file_ytrain = open('ytrain.txt', 'w')
file_Xtest  = open('Xtest.txt', 'w')
file_ytest  = open('ytest.txt', 'w')

### Parameters needed to show animation of spin configurations: ###
animate = False
bw_cmap = colors.ListedColormap(['black', 'white'])

### Initially, the spins are in a random state (a high-T phase): ###
spins = np.zeros(N_spins,dtype=np.int)
for i in range(N_spins):
  spins[i] = 2*random.randint(0,1) - 1 #either +1 or -1

### Store each spin's four nearest neighbours in a neighbours array (using periodic boundary conditions): ###
neighbours = np.zeros((N_spins,4),dtype=np.int)
for i in range(N_spins):
  #neighbour to the right:
  neighbours[i,0]=i+1
  if i%L==(L-1):
    neighbours[i,0]=i+1-L
  
  #upwards neighbour:
  neighbours[i,1]=i+L
  if i >= (N_spins-L):
    neighbours[i,1]=i+L-N_spins
  
  #neighbour to the left:
  neighbours[i,2]=i-1
  if i%L==0:
    neighbours[i,2]=i-1+L
  
  #downwards neighbour:
  neighbours[i,3]=i-L
  if i <= (L-1):
    neighbours[i,3]=i-L+N_spins
#end of for loop

### Function to calculate the total energy ###
def getEnergy():
  currEnergy = 0
  for i in range(N_spins):
    currEnergy += -J*( spins[i]*spins[neighbours[i,0]] + spins[i]*spins[neighbours[i,1]] )
  return currEnergy
#end of getEnergy() function

### Function to calculate the total magnetization ###
def getMag():
  return np.sum(spins)
#end of getMag() function

### Function to perform one Monte Carlo sweep ###
def sweep():
  #do one sweep (N_spins local updates):
  for i in range(N_spins):
    #randomly choose which spin to consider flipping:
    site = random.randint(0,N_spins-1)
      
    deltaE = 0
    #calculate the change in energy of the proposed move by considering only the nearest neighbours:
    for j in range(4):
      deltaE += 2*J*spins[site]*spins[neighbours[site,j]]
  
    if (deltaE <= 0) or (random.random() < np.exp(-deltaE/T)):
      #flip the spin:
      spins[site] = -spins[site]
  #end loop over i
#end of sweep() function

### Function to write the training/testing data to file: ###
def writeConfigs(num,T):
  #determine whether the current configuration will be used for training or testing:
  if num < (train_frac*n_bins):
    file_X = file_Xtrain
    file_y = file_ytrain
  else:
    file_X = file_Xtest
    file_y = file_ytest

  #multiply the configuration by +1 or -1 to ensure we generate configurations with both positive and negative magnetization:
  flip = 2*random.randint(0,1) - 1
    
  #loop to write each spin to a single line of the X data file:
  for i in range(N_spins):
    currSpin = flip*spins[i]
      
    #replace -1 with 0 (to be consistent with the desired format):
    if currSpin == -1:
      currSpin = 0
    
    file_X.write('%d  ' %(currSpin) )
  #end loop over i
  file_X.write('\n')
  
  y = 0
  if T>Tc:
    y = 1
  file_y.write('%d \n' %y)
#end of writeConfigs(num,T) function

#################################################################################
########## Loop over all temperatures and perform Monte Carlo updates: ##########
#################################################################################
for T in T_list:
  print('\nT = %f' %T)
  
  #open a file where observables will be recorded:
  fileName         = 'ising2d_L%d_T%.4f.txt' %(L,T)
  file_observables = open(fileName, 'w')
  
  #equilibration sweeps:
  for i in range(n_eqSweeps):
    sweep()

  #start doing measurements:
  for i in range(n_bins):
    for j in range(n_sweepsPerBin):
      sweep()
    #end loop over j

    #Write the observables to file:
    energy = getEnergy()
    mag    = getMag()
    file_observables.write('%d \t %.8f \t %.8f \n' %(i, energy, mag))

    #write the X, y data to file:
    writeConfigs(i,T)

    if animate:
      #Display the current spin configuration:
      plt.clf()
      plt.imshow( spins.reshape((L,L)), cmap=bw_cmap, norm=colors.BoundaryNorm([-1,0,1], bw_cmap.N) )
      plt.xticks([])
      plt.yticks([])
      plt.title('%d x %d Ising model, T = %.3f' %(L,L,T))
      plt.pause(0.01)
    #end if

    if (i+1)%50==0:
      print ('  %d bins complete' %(i+1))
  #end loop over i

  file_observables.close()
#end loop over temperature
