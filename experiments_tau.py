import seaborn as sns
from main import main
import numpy as np
import matplotlib.pyplot as plt
import random
import math
import pandas as pd


# =============================================================================
# First we want to understand the behavior of main() during a run, before modifying the network structure.
# A/ How does fitness changes during a run?
# B/ How does very different networks appear to change the trajectory?
# C/ How does Constant vs Redrawn networks changes fitness?
# =============================================================================

def experiment_curve_tau(rep,tau,y,p,x,z):
  random.seed(9)
  final_results = np.zeros((rep,tau))
  time_axis = list(range(0,tau))
  r = 0
  while r < rep:
    mean_fitness, fitness_history, time_history, A = main(y, p,tau,x,z)
    final_results[r,] = np.copy(fitness_history)
    print(r)
    r += 1
  return final_results, time_axis

# =============================================================================
# Experiments programme
  # 1- ER network, p=0, tau=100, cons
  # 1b- ER network, p=0, tau=100, dyn (optional)
  # 2- ER network, p=0.5, tau=100, cons
  # 2b- ER network, p=0.5, tau=100, dyn (optional)
  # 3- ER network, p=1, tau=100, cons
  # 3b- ER network, p=1, tau=100, dyn (optional)
  # 4- AB network, m=1, tau=100, cons
  # 4b- AB network, m=1, tau=100, dyn (optional)
  # 5- AB network, m=25, tau=100, cons
  # 5b- AB network, m=25, tau=100, dyn (optional)
  # 6- AB network, m=n-1, tau=100, cons
  # 6b- AB network, m=n-1, tau=100, dyn (optional)
# =============================================================================
  
# =============================================================================
#   Numerical experiments to get the data
# =============================================================================
  
final_results_sphere_1, time_axis = experiment_curve_tau(10,100,'sphere',0,'erdos','cons')
final_results_ackley_1, time_axis = experiment_curve_tau(10,100,'ackley',0,'erdos','cons')
final_results_rastrigin_1, time_axis = experiment_curve_tau(10,100,'rastrigin',0,'erdos','cons')

final_results_sphere_2, time_axis = experiment_curve_tau(10,100,'sphere',0.5,'erdos','cons')
final_results_ackley_2, time_axis = experiment_curve_tau(10,100,'ackley',0.5,'erdos','cons')
final_results_rastrigin_2, time_axis = experiment_curve_tau(10,100,'rastrigin',0.5,'erdos','cons')

final_results_sphere_3, time_axis = experiment_curve_tau(10,100,'sphere',1,'erdos','cons')
final_results_ackley_3, time_axis = experiment_curve_tau(10,100,'ackley',1,'erdos','cons')
final_results_rastrigin_3, time_axis = experiment_curve_tau(10,100,'rastrigin',1,'erdos','cons')

final_results_sphere_4, time_axis = experiment_curve_tau(10,100,'sphere',1,'albert','cons')
final_results_ackley_4, time_axis = experiment_curve_tau(10,100,'ackley',1,'albert','cons')
final_results_rastrigin_4, time_axis = experiment_curve_tau(10,100,'rastrigin',1,'albert','cons')

final_results_sphere_5, time_axis = experiment_curve_tau(10,100,'sphere',25,'albert','cons')
final_results_ackley_5, time_axis = experiment_curve_tau(10,100,'ackley',25,'albert','cons')
final_results_rastrigin_5, time_axis = experiment_curve_tau(10,100,'rastrigin',25,'albert','cons')

final_results_sphere_6, time_axis = experiment_curve_tau(10,100,'sphere',49,'albert','cons')
final_results_ackley_6, time_axis = experiment_curve_tau(10,100,'ackley',49,'albert','cons')
final_results_rastrigin_6, time_axis = experiment_curve_tau(10,100,'rastrigin',49,'albert','cons')

""" Intermediate raw data """ 

sphere1 = pd.DataFrame(final_results_sphere_1)
sphere1.to_csv('sphere1.csv')
sphere2 = pd.DataFrame(final_results_sphere_2)
sphere2.to_csv('sphere2.csv')
sphere3 = pd.DataFrame(final_results_sphere_3)
sphere3.to_csv('sphere3.csv')
sphere4 = pd.DataFrame(final_results_sphere_4)
sphere4.to_csv('sphere4.csv')
sphere5 = pd.DataFrame(final_results_sphere_5)
sphere5.to_csv('sphere5.csv')
sphere6 = pd.DataFrame(final_results_sphere_6)
sphere6.to_csv('sphere6.csv')

ackley1 = pd.DataFrame(final_results_ackley_1)
ackley1.to_csv('ackley1.csv')
ackley2 = pd.DataFrame(final_results_ackley_2)
ackley2.to_csv('ackley2.csv')
ackley3 = pd.DataFrame(final_results_ackley_3)
ackley3.to_csv('ackley3.csv')
ackley4 = pd.DataFrame(final_results_ackley_4)
ackley4.to_csv('ackley4.csv')
ackley5 = pd.DataFrame(final_results_ackley_5)
ackley5.to_csv('ackley5.csv')
ackley6 = pd.DataFrame(final_results_ackley_6)
ackley6.to_csv('ackley6.csv')

rastrigin1 = pd.DataFrame(final_results_rastrigin_1)
rastrigin1.to_csv('rastrigin1.csv')
rastrigin2 = pd.DataFrame(final_results_rastrigin_2)
rastrigin2.to_csv('rastrigin2.csv')
rastrigin3 = pd.DataFrame(final_results_rastrigin_3)
rastrigin3.to_csv('rastrigin3.csv')
rastrigin4 = pd.DataFrame(final_results_rastrigin_4)
rastrigin4.to_csv('rastrigin4.csv')
rastrigin5 = pd.DataFrame(final_results_rastrigin_5)
rastrigin5.to_csv('rastrigin5.csv')
rastrigin6 = pd.DataFrame(final_results_rastrigin_6)
rastrigin6.to_csv('rastrigin6.csv')


""" Processed data (means) """ 

df = pd.DataFrame()
df['sphere1'] =  np.mean(final_results_sphere_1,axis=0)
df['sphere2'] = np.mean(final_results_sphere_2,axis=0)
df['sphere3'] = np.mean(final_results_sphere_3,axis=0)
df['sphere4'] = np.mean(final_results_sphere_4,axis=0)
df['sphere5'] = np.mean(final_results_sphere_5,axis=0)
df['sphere6'] = np.mean(final_results_sphere_6,axis=0)

df['ackley1'] = np.mean(final_results_ackley_1,axis=0)
df['ackley2'] = np.mean(final_results_ackley_2,axis=0)
df['ackley3'] = np.mean(final_results_ackley_3,axis=0)
df['ackley4'] = np.mean(final_results_ackley_4,axis=0)
df['ackley5'] = np.mean(final_results_ackley_5,axis=0)
df['ackley6'] = np.mean(final_results_ackley_6,axis=0)

df['rastrigin1'] = np.mean(final_results_rastrigin_1,axis=0)
df['rastrigin2'] = np.mean(final_results_rastrigin_2,axis=0)
df['rastrigin3'] = np.mean(final_results_rastrigin_3,axis=0)
df['rastrigin4'] = np.mean(final_results_rastrigin_4,axis=0)
df['rastrigin5'] = np.mean(final_results_rastrigin_5,axis=0)
df['rastrigin6'] = np.mean(final_results_rastrigin_6,axis=0)

df.to_csv('data_tau.csv')

# =============================================================================
#   Plotting the results
# =============================================================================
  
fig, ((ax, ax2,ax3),(ax4,ax5,ax6))  = plt.subplots(2, 3, sharey=True, sharex = True , figsize=(11,5))

ax.plot(time_axis, np.mean(final_results_sphere_1,axis=0), label = "Sphere function", color = "red")
ax.plot(time_axis, np.mean(final_results_ackley_1,axis=0), label = "Ackley function", color = "blue")
ax.plot(time_axis, np.mean(final_results_rastrigin_1,axis=0), label = "Rastrigin function", color = "purple")
ax.set_xlabel('t')
ax.set_ylabel('Avg. fitness')
ax.set_title('ER network with $p=0$')

ax2.plot(time_axis, np.mean(final_results_sphere_2,axis=0), label = "Sphere function", color = "red")
ax2.plot(time_axis, np.mean(final_results_ackley_2,axis=0), label = "Ackley function", color = "blue")
ax2.plot(time_axis, np.mean(final_results_rastrigin_2,axis=0), label = "Rastrigin function", color = "purple")
ax2.set_xlabel('t')
ax2.set_ylabel('Avg. fitness')
ax2.set_title('ER network with $p=0.5$')

ax3.plot(time_axis, np.mean(final_results_sphere_3,axis=0), label = "Sphere function", color = "red")
ax3.plot(time_axis, np.mean(final_results_ackley_3,axis=0), label = "Ackley function", color = "blue")
ax3.plot(time_axis, np.mean(final_results_rastrigin_3,axis=0), label = "Rastrigin function", color = "purple")
ax3.set_xlabel('t')
ax3.set_ylabel('Avg. fitness')
ax3.set_title('ER network with $p=1$')

ax4.plot(time_axis, np.mean(final_results_sphere_4,axis=0), label = "Sphere function", color = "red")
ax4.plot(time_axis, np.mean(final_results_ackley_4,axis=0), label = "Ackley function", color = "blue")
ax4.plot(time_axis, np.mean(final_results_rastrigin_4,axis=0), label = "Rastrigin function", color = "purple")
ax4.set_xlabel('t')
ax4.set_ylabel('Avg. fitness')
ax4.set_title('AB network with $m=1$')

ax5.plot(time_axis, np.mean(final_results_sphere_5,axis=0), label = "Sphere function", color = "red")
ax5.plot(time_axis, np.mean(final_results_ackley_5,axis=0), label = "Ackley function", color = "blue")
ax5.plot(time_axis, np.mean(final_results_rastrigin_5,axis=0), label = "Rastrigin function", color = "purple")
ax5.set_xlabel('t')
ax5.set_ylabel('Avg. fitness')
ax5.set_title('AB network with $m=25$')

ax6.plot(time_axis, np.mean(final_results_sphere_6,axis=0), label = "Sphere function", color = "red")
ax6.plot(time_axis, np.mean(final_results_ackley_6,axis=0), label = "Ackley function", color = "blue")
ax6.plot(time_axis, np.mean(final_results_rastrigin_6,axis=0), label = "Rastrigin function", color = "purple")
ax6.set_xlabel('t')
ax6.set_ylabel('Avg. fitness')
ax6.set_title('AB network with $m=49$')
handles, labels = ax6.get_legend_handles_labels()
lg = fig.legend(handles, labels,fancybox=False, shadow=False, loc='center',ncol=3, bbox_to_anchor=(0.45, 0.04))
plt.savefig("tau_behavior.png", format="png",dpi=300)
plt.show()
