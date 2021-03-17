# =============================================================================
# Imports
# =============================================================================
import seaborn as sns

# =============================================================================
# Define the experimentation function, i.e. a function that will perform a series
# of experiments to analyse the behavior of main(), wrt p
# main (function name, network parameter, number of iterations, network type, network status)
# =============================================================================

def experiment_curve_p(rep,tau,y,pstart,pend,pincr,x,z):
  random.seed(9)
  p = pstart
  i = 0
  final_results = np.zeros((math.ceil((pend-pstart)*1/pincr),rep))
  p_axis = []
  while p <= pend:
    print(np.round(p,2))
    r = 0
    while r < rep:
      mean_fitness, fitness_history, time_history, A  = main(y, p,tau,x,z)
      final_results[i,r] = mean_fitness
      r += 1
    p_axis.append(p)
    p += pincr
    i += 1
  return final_results, p_axis, fitness_history, time_history

# =============================================================================
# Experiments programme
  # 1- ER network, p in [0,1,0.01], tau=20, cons
  # 1b- ER network, p in [0,1,0.01], tau=20, dyn (optional)
  # 2- ER network, p in [0,1,0.01], tau=50, cons
  # 2b- ER network, p in [0,1,0.01], tau=50, dyn (optional)
  # 3- ER network, p in [0,1,0.01], tau=100, cons
  # 3b- ER network, p in [0,1,0.01], tau=100, dyn (optional)
  # 4- AB network, m in [1,49,1], tau=20, cons
  # 4b- AB network, m in [1,49,1], tau=20, dyn (optional)
  # 5- AB network, m in [1,49,1], tau=50, cons
  # 5b- AB network, m in [1,49,1], tau=50, dyn (optional)
  # 6- AB network, m in [1,49,1], tau=100, cons
  # 6b- AB network, m in [1,49,1], tau=100, dyn (optional)
  
# =============================================================================
#   Numerical experiments to get the data for Erdos
# =============================================================================
  
ackley_results_1, p_axis, fitness_history_sphere, time_history = experiment_curve_p(10,20,ackley,0,1.01,0.01,erdos,cons) 
ackley_results_2, p_axis, fitness_history_sphere, time_history = experiment_curve_p(10,50,ackley,0,1.01,0.01,erdos,cons) 
ackley_results_3, p_axis, fitness_history_sphere, time_history = experiment_curve_p(10,100,ackley,0,1.01,0.01,erdos,cons) 

rastrigin_results_1, p_axis, fitness_history_sphere, time_history = experiment_curve_p(10,20,rastrigin,0,1.01,0.01,erdos,cons) 
rastrigin_results_2, p_axis, fitness_history_sphere, time_history = experiment_curve_p(10,50,rastrigin,0,1.01,0.01,erdos,cons) 
rastrigin_results_3, p_axis, fitness_history_sphere, time_history = experiment_curve_p(10,100,rastrigin,0,1.01,0.01,erdos,cons) 

sphere_results_1, p_axis, fitness_history_sphere, time_history = experiment_curve_p(10,20,sphere,0,1.01,0.01,erdos,cons) 
sphere_results_2, p_axis, fitness_history_sphere, time_history = experiment_curve_p(10,50,sphere,0,1.01,0.01,erdos,cons) 
sphere_results_3, p_axis, fitness_history_sphere, time_history = experiment_curve_p(10,100,sphere,0,1.01,0.01,erdos,cons) 

# =============================================================================
#   Plotting the results for ERDOS
# =============================================================================
fig, ((ax, ax2,ax3),(ax4,ax5,ax6),(ax7,ax8,ax9)) = plt.subplots(3, 3, sharey='row', sharex = True , figsize=(11,5))
# 1st
sns.regplot(p_axis,np.mean(rastrigin_results_1,axis=1),order=4, ci=95,line_kws={"color": "red"}, scatter_kws={'s':2,"color": "blue"}, ax=ax, truncate = True)
ax.set_ylabel('Rastrigin function')
ax.set_title('$t=20$')
# 2nd
sns.regplot(p_axis,np.mean(rastrigin_results_2,axis=1),order=4, ci=95,line_kws={"color": "red"}, scatter_kws={'s':2,"color": "blue"}, ax=ax2, truncate = True)
ax2.set_title('$t=50$')
# 3rd
sns.regplot(p_axis,np.mean(rastrigin_results_3,axis=1),order=4, ci=95,line_kws={"color": "red"}, scatter_kws={'s':2,"color": "blue"}, ax=ax3, truncate = True)
ax3.set_title('$t=100$')
#4th
sns.regplot(p_axis,np.mean(sphere_results_1,axis=1),order=4, ci=95,line_kws={"color": "red"}, scatter_kws={'s':2,"color": "blue"}, ax=ax4, truncate = True)
ax4.set_ylabel('Sphere function')
# 5th
sns.regplot(p_axis,np.mean(sphere_results_2,axis=1),order=4, ci=95,line_kws={"color": "red"}, scatter_kws={'s':2,"color": "blue"}, ax=ax5, truncate = True)
# 6th
sns.regplot(p_axis,np.mean(sphere_results_3,axis=1),order=4, ci=95,line_kws={"color": "red"}, scatter_kws={'s':2,"color": "blue"}, ax=ax6, truncate = True)
# 7th
sns.regplot(p_axis,np.mean(ackley_results_1,axis=1),order=4, ci=95,line_kws={"color": "red"}, scatter_kws={'s':2,"color": "blue"}, ax=ax7, truncate = True)
ax7.set_ylabel('Ackley function')
ax7.set_xlabel('$p$')
# 8th
sns.regplot(p_axis,np.mean(ackley_results_2,axis=1),order=4, ci=95,line_kws={"color": "red"}, scatter_kws={'s':2,"color": "blue"}, ax=ax8, truncate = True)
ax8.set_xlabel('$p$')
# 9th
sns.regplot(p_axis,np.mean(ackley_results_3,axis=1),order=4, ci=95,line_kws={"color": "red"}, scatter_kws={'s':2,"color": "blue"}, ax=ax9, truncate = True)
ax9.set_xlabel('$p$')
# General
fig.suptitle(' ER network ')
plt.savefig("results_erdos.png", format="png",dpi=300)
plt.show()

# =============================================================================
#   Numerical experiments to get the data for Albert
# =============================================================================
  
ackley_results_4, p_axis, fitness_history_ackley_4, time_history = experiment_curve_p(10,20,ackley,1,49.01,1,albert,cons) 
ackley_results_5, p_axis, fitness_history_ackley_5,, time_history = experiment_curve_p(10,50,ackley,1,49.01,1,albert,cons) 
ackley_results_6, p_axis, fitness_history_ackley_6,, time_history = experiment_curve_p(10,100,ackley,1,49.01,1,albert,cons) 

rastrigin_results_4, p_axis, fitness_history_sphere, time_history = experiment_curve_p(10,20,rastrigin,1,49.01,1,albert,cons) 
rastrigin_results_5, p_axis, fitness_history_sphere, time_history = experiment_curve_p(10,50,rastrigin,1,49.01,1,albert,cons)  
rastrigin_results_6, p_axis, fitness_history_sphere, time_history = experiment_curve_p(10,100,rastrigin,1,49.01,1,albert,cons) 

sphere_results_4, p_axis, fitness_history_sphere, time_history = experiment_curve_p(10,20,sphere,1,49.01,1,albert,cons)  
sphere_results_5, p_axis, fitness_history_sphere, time_history = experiment_curve_p(10,50,sphere,1,49.01,1,albert,cons) 
sphere_results_6, p_axis, fitness_history_sphere, time_history = experiment_curve_p(10,100,sphere,1,49.01,1,albert,cons)

# =============================================================================
#   Plotting the results for ALBERT
# =============================================================================
fig, ((ax, ax2,ax3),(ax4,ax5,ax6),(ax7,ax8,ax9)) = plt.subplots(3, 3, sharey='row', sharex = True , figsize=(11,5))
# 1st
sns.regplot(p_axis,np.mean(rastrigin_results_4,axis=1),order=4, ci=95,line_kws={"color": "red"}, scatter_kws={'s':2,"color": "blue"}, ax=ax, truncate = True)
ax.set_ylabel('Rastrigin function')
ax.set_title('$t=20$')
# 2nd
sns.regplot(p_axis,np.mean(rastrigin_results_5,axis=1),order=4, ci=95,line_kws={"color": "red"}, scatter_kws={'s':2,"color": "blue"}, ax=ax2, truncate = True)
ax2.set_title('$t=50$')
# 3rd
sns.regplot(p_axis,np.mean(rastrigin_results_6,axis=1),order=4, ci=95,line_kws={"color": "red"}, scatter_kws={'s':2,"color": "blue"}, ax=ax3, truncate = True)
ax3.set_title('$t=100$')
#4th
sns.regplot(p_axis,np.mean(sphere_results_4,axis=1),order=4, ci=95,line_kws={"color": "red"}, scatter_kws={'s':2,"color": "blue"}, ax=ax4, truncate = True)
ax4.set_ylabel('Sphere function')
# 5th
sns.regplot(p_axis,np.mean(sphere_results_5,axis=1),order=4, ci=95,line_kws={"color": "red"}, scatter_kws={'s':2,"color": "blue"}, ax=ax5, truncate = True)
# 6th
sns.regplot(p_axis,np.mean(sphere_results_6,axis=1),order=4, ci=95,line_kws={"color": "red"}, scatter_kws={'s':2,"color": "blue"}, ax=ax6, truncate = True)
# 7th
sns.regplot(p_axis,np.mean(ackley_results_4,axis=1),order=4, ci=95,line_kws={"color": "red"}, scatter_kws={'s':2,"color": "blue"}, ax=ax7, truncate = True)
ax7.set_ylabel('Ackley function')
ax7.set_xlabel('$m$')
# 8th
sns.regplot(p_axis,np.mean(ackley_results_5,axis=1),order=4, ci=95,line_kws={"color": "red"}, scatter_kws={'s':2,"color": "blue"}, ax=ax8, truncate = True)
ax8.set_xlabel('$m$')
# 9th
sns.regplot(p_axis,np.mean(ackley_results_6,axis=1),order=4, ci=95,line_kws={"color": "red"}, scatter_kws={'s':2,"color": "blue"}, ax=ax9, truncate = True)
ax9.set_xlabel('$m$')
# General
fig.suptitle('AB network')
plt.savefig("results_albert.png", format="png",dpi=300)
plt.show()


# =============================================================================
# Data for the results table 
# What we are recording:
# For each test function, at the three terminal times considered:
# - The best fitness scores, averaged across the 10 runs, across different all networks for the ER and AB type
# - Thee average fitness score of the traditional "complete" GA
# =============================================================================

# rastrigin
rastrigin_avg20_ER = np.round(np.min(np.mean(rastrigin_results_1,axis=1)),3)
rastrigin_avg50_ER = np.round(np.min(np.mean(rastrigin_results_2,axis=1)),3)
rastrigin_avg100_ER = np.round(np.min(np.mean(rastrigin_results_3,axis=1)),3)
rastrigin_avg20_AB = np.round(np.min(np.mean(rastrigin_results_4,axis=1)),3)
rastrigin_avg50_AB = np.round(np.min(np.mean(rastrigin_results_5,axis=1)),3)
rastrigin_avg100_AB = np.round(np.min(np.mean(rastrigin_results_6,axis=1)),3)
rastrigin_avg20_GA = np.round(np.mean(rastrigin_results_1,axis=1)[-1],3)
rastrigin_avg50_GA = np.round(np.mean(rastrigin_results_2,axis=1)[-1],3)
rastrigin_avg100_GA = np.round(np.mean(rastrigin_results_3,axis=1)[-1],3)

# sphere
sphere_avg20_ER = np.round(np.min(np.mean(sphere_results_1,axis=1)),3)
sphere_avg50_ER = np.round(np.min(np.mean(sphere_results_2,axis=1)),3)
sphere_avg100_ER = np.round(np.min(np.mean(sphere_results_3,axis=1)),3)
sphere_avg20_AB = np.round(np.min(np.mean(sphere_results_4,axis=1)),3)
sphere_avg50_AB = np.round(np.min(np.mean(sphere_results_5,axis=1)),3)
sphere_avg100_AB = np.round(np.min(np.mean(sphere_results_6,axis=1)),3)
sphere_avg20_GA = np.round(np.mean(sphere_results_1,axis=1)[-1],3)
sphere_avg50_GA = np.round(np.mean(sphere_results_2,axis=1)[-1],3)
sphere_avg100_GA = np.round(np.mean(sphere_results_3,axis=1)[-1],3)

# ackley
ackley_avg20_ER = np.round(np.min(np.mean(ackley_results_1,axis=1)),3)
ackley_avg50_ER = np.round(np.min(np.mean(ackley_results_2,axis=1)),3)
ackley_avg100_ER = np.round(np.min(np.mean(ackley_results_3,axis=1)),3)
ackley_avg20_AB = np.round(np.min(np.mean(ackley_results_4,axis=1)),3)
ackley_avg50_AB = np.round(np.min(np.mean(ackley_results_5,axis=1)),3)
ackley_avg100_AB = np.round(np.min(np.mean(ackley_results_6,axis=1)),3)
ackley_avg20_GA = np.round(np.mean(ackley_results_1,axis=1)[-1],3)
ackley_avg50_GA = np.round(np.mean(ackley_results_2,axis=1)[-1],3)
ackley_avg100_GA = np.round(np.mean(ackley_results_3,axis=1)[-1],3)


