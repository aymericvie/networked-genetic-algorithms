# Networked Genetic Algorithm (NGA)

While all genetic algorithms commonly allow each individual in the population to mate with any other individuals, nature and social systems do not exhibit such a complete network structure. The NGA is an attempt to identify how the network structure of the population impacts the performance of genetic algorithms.
To do this, I use NGAs to optimise some test functions, and run various network structures to identify these effects.
The results identify such an effect, which appears significant. In addition, the best performing NGA outperforms the standard GA, in average reducing fitness scores by 53%.

This repository contains all the code involved in this research project. 

## How to use this code

- test_functions_landscape defines the test functions and plots their landscapes with pretty colors
- landscapes_united creates a combined plot for all test functions landscapes
- network_features creates plenty of networks, plots different examples to illustrate how network structure changes with respect to the network structure parameters. It also shows how some key features of networks change with those parameters.
- main includes all the code to run the NGA, in the form of the main() function.
- experiments_tau allows to perform robust experiments on the behavior of the NGA over time
- experiments_random_network_structure allows to perform robust experiments on the behavior of the NGA with respect to network structure

The paper is linked here: https://arxiv.org/abs/2104.04254. This research has been accepted to the conference GECCO 2021.

## Reproducibility and data

For reproducibility purposes, the intermediate and final data of the analysis has been published. The folder data_tau contains the data on the performance of the NGAs over time. The folder data_nw contains the data on the performance of the NGAs depending on network structure.

## Computational cost
The experiments run in 30 to 90 minutes on a commercial laptop (Intel Core i5-10310U CPU 1.7GHz, 16BG RAM).
Computation uses standard packages: math, random, numpy, matplotlib.

## Acknowledgments
Many thanks for reproduciblity advice and help to Anja Jankovic, Andreea Avramescu, Lennart Schapermeier, Manuel Lopez-Ibanez
