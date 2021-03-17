# =============================================================================
# # Imports
# =============================================================================
import matplotlib 
matplotlib.rcParams['pdf.fonttype'] = 42 
matplotlib.rcParams['ps.fonttype'] = 42
import numpy as np
np.set_printoptions(precision=3, suppress=True)
import random
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import networkx as nx
from networkx.drawing.nx_pydot import graphviz_layout
import math
random.seed(9)

# =============================================================================
# # Test functions
# =============================================================================

def rastrigin(x, y):
    return 20 + x**2 - 10*np.cos(2*np.pi*x) + y**2 - 10*np.cos(2*np.pi*y)
def sphere(x, y):
    return x**2 + y**2
def ackley(x, y):
    return -20*np.exp(-0.2*np.sqrt(0.5*(x**2+y**2))) - np.exp(0.5*(np.cos(2*np.pi*x) + np.cos(2*np.pi*y))) + 20 + np.exp(1)

# =============================================================================
# # Main GA fixed parameters
# =============================================================================

d = 2 #dimensionality of the test functions
n = 50 #population size        
rho = 0.7 #crossover rate
mu = 0.05 #mutation rate       
tau = 1000 #number of iterations  ### TBC 1000
p = 0 #link probability
erdos = 0 #network type
albert = 1 #network type
dyn = 1 #network draw type
cons = 0 #network draw type

# Cons = the nw structure is a once-for-all draw at the initialisation. 
# Dyn = The nw structure is drawn from the corresponding parameter at every round

def main(f,p,tau,x,z):

  # determine search domain depending on the selected test function
  if f == ackley:
    upperb = 32.768
    lowerb = -32.768
  if f == sphere:
    upperb = 5.12
    lowerb = -5.12
  if f == rastrigin:
    upperb = 5.12
    lowerb = -5.12

  # determine function to optimise wrt selected test function
  def func(f,x,y):
    if f == ackley:
      return ackley(x,y)
    if f == sphere:
      return sphere(x,y)
    if f == rastrigin:
      return rastrigin(x,y)

  # Initial sampling of population, and definition of some results arrays
  population  = np.random.uniform(lowerb,upperb,(n,d))
  fitness_history = []
  time_history = []
  global A
  A = []

  if z == cons:

      # Determine Population network - adjacency matrix of the corresponding random network
      if x == erdos:
          A = nx.adjacency_matrix(nx.erdos_renyi_graph(n, p, seed=None, directed=False)).todense()
          # p in [0,1] by 0.01
      if x == albert:
          A = nx.adjacency_matrix(nx.barabasi_albert_graph(n, p, seed=None)).todense()
          # p in [0,n-1] by 1
  t = 0
  while t < tau:
    # If network is dynamic, draw the time period network
    if z == dyn:
      # Determine Population network - adjacency matrix of the corresponding random network
      
      if x == erdos:
            A = nx.adjacency_matrix(nx.erdos_renyi_graph(n, p, seed=None, directed=False)).todense()
            # p in [0,1] by 0.01
      if x == albert:
            A = nx.adjacency_matrix(nx.barabasi_albert_graph(n, p, seed=None)).todense()
            # p in [0,n-1] by 
          
    # Compute general fitness and fitness-proportionate selection probabilities
    fitness = np.zeros((n,1))
    i = 0
    while i < n:
      fitness[i,0] = func(f, population[i,0], population[i,1])
      i += 1
    adjusted_fitness = 1/(1+fitness)
    selection_proba = adjusted_fitness / np.sum(adjusted_fitness)
    cumulative_fitness = np.copy(selection_proba)
    i = 1
    while i < n:
      cumulative_fitness[i,0] += cumulative_fitness[i-1,0]
      i += 1
    
    # iterate selection and reproduction until we have a full next generation (requires an even population size)
    v = 0
    next_population = np.zeros((1,d))
    while v < n:
      # Fitness-proportionate selection in the full population
      a = random.uniform(0,1)
      a_count = 0
      while a > cumulative_fitness[a_count]:
        a_count += 1

      parent1 = np.copy(population[a_count,])
      parent1 = np.reshape(parent1, (1,2))

      # Fitness-proportionate selection in the neighborhood of parent 1, as many times as needed to create n new offspring

      # if no node is connected to parent1, it will replicate itself
      if np.sum(A,axis=1)[a_count] == 0:
        offspring1 = np.copy(parent1)
        offspring2 = np.copy(parent1)

      # if at least some node is connected to parent1, operate fitness-proportionate selection in this neighborhood
      if np.sum(A,axis=1)[a_count] != 0:
        fitness_neighbor = np.copy(fitness)
        count = 0
        i = 0
        while i < n:
          if A[a_count,i] == 0: #if we have no connection to individual i, i cannot be selected.
          # This works also to prevent self loops: an individual cannot select itself 
            fitness_neighbor[i,0] = float("inf")
            count += 1
          i+= 1

        adjusted_fitness_neighbor = 1/(1+fitness_neighbor)
        selection_proba_neighbor = adjusted_fitness_neighbor / np.sum(adjusted_fitness_neighbor)
        cumulative_fitness_neighbor = np.copy(selection_proba_neighbor)
        i = 1
        while i < n:
          cumulative_fitness_neighbor[i,0] += cumulative_fitness_neighbor[i-1,0]
          i += 1
        b = random.uniform(0,1)
        b_count = 0
        while b > cumulative_fitness_neighbor[b_count]:
          b_count += 1
        parent2 = np.copy(population[b_count,])
        parent2 = np.reshape(parent2, (1,2))

        # Do crossover between parent 1 and parent 2 with proba rho
        m = random.uniform(0,1)
        if m <= rho:
          # Do crossover between parent 1 and parent 2
          
          offspring1 = np.zeros((1, d))
          offspring2 = np.zeros((1, d))
          cpoint = random.randint(0,d)
          j = 0
          while j < cpoint:
            offspring1[0,j] = parent1[0,j]
            offspring2[0,j] = parent2[0,j]
            j += 1
          while j < d:
            offspring1[0,j] = parent2[0,j]
            offspring2[0,j] = parent1[0,j]
            j += 1

        elif m > rho:
          # # Self replication
          offspring1 = np.copy(parent1)
          offspring2 = np.copy(parent2)

      # Do mutation with proba mu
      i = 0
      while i < d:
        m = random.uniform(0,1)
        if m <= mu: #nothing happens otherwise
          offspring1[0,i] = offspring1[0,i] + np.random.normal(0,1)# mutate
        i += 1

      # Add the offspring to the next population
      next_population = np.vstack((next_population, offspring1))
      next_population = np.vstack((next_population, offspring2))
      del offspring1
      del offspring2
      v+= 2

    # Record results
    fitness_history.append(np.mean(fitness))
    time_history.append(t)
    
    next_population = np.delete(next_population, 0, 0)
    population = next_population
    t += 1
  
  return np.mean(fitness), fitness_history, time_history, A

# =============================================================================
# Testing
# =============================================================================
  
#main(sphere, 1, 100, erdos, cons)
#main(sphere, 1, 100, erdos, dyn)
