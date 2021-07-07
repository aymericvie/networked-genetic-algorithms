# main(f,p,tau,x,z)
# main (function name, network parameter, number of iterations, network type, network status)
rep = 100

r = 0
sum_mean_fitness = 0
while r < rep:
    mean_fitness, fitness_history, time_history, A = main(sphere,1,20,erdos,cons)
    sum_mean_fitness += mean_fitness
    r += 1
mean_mean_fitness = sum_mean_fitness / (r+1)
print("avg fitness for p1")
print(mean_mean_fitness)

r = 0
sum_mean_fitness = 0
while r < rep:
    mean_fitness, fitness_history, time_history, A = main(sphere,0,20,erdos,cons)
    sum_mean_fitness += mean_fitness
    r += 1
mean_mean_fitness = sum_mean_fitness / (r+1)
print("avg fitness for p0")
print(mean_mean_fitness)

#############

r = 0
sum_mean_fitness = 0
while r < rep:
    mean_fitness, fitness_history, time_history, A = main(ackley,1,20,erdos,cons)
    sum_mean_fitness += mean_fitness
    r += 1
mean_mean_fitness = sum_mean_fitness / (r+1)
print("avg fitness for p1")
print(mean_mean_fitness)

r = 0
sum_mean_fitness = 0
while r < rep:
    mean_fitness, fitness_history, time_history, A = main(ackley,0,20,erdos,cons)
    sum_mean_fitness += mean_fitness
    r += 1
mean_mean_fitness = sum_mean_fitness / (r+1)
print("avg fitness for p0")
print(mean_mean_fitness)
