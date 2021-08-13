# imports
import sys
from matplotlib import pyplot as plt
import os
import numpy as np
# add folder to syspath to import GeneticAlgorithm 
path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(path)+'/GeneticAlgorithm')
from GeneticAlgorithm import GeneticAlgorithm
# add folder to syspath to import ParticleSwarmOptimization
sys.path.insert(0, os.path.dirname(path)+'/ParticleSwarmOptimization')
 # add folder to syspath to import ACO 
sys.path.insert(0, os.path.dirname(path)+'/AntColonyOptimization')

# load the different Problems
# small problem
small_objects  = np.load(os.path.join(path,'Ressources/betterproblem/small_objects.npy'))
small_container = np.load(os.path.join(path,'Ressources/betterproblem/small_container.npy'))
small_optimal_solution = np.load(os.path.join(path, 'Ressources/betterproblem/small_optimal.npy'))
# medium Problem
medium_objects= np.load(os.path.join(path,'Ressources/betterproblem/medium_objects.npy'))
medium_container = np.load(os.path.join(path,'Ressources/betterproblem/medium_container.npy'))
medium_optimal_solution = np.load(os.path.join(path, 'Ressources/betterproblem/medium_optimal.npy'))
# large Problem
large_objects  = np.load(os.path.join(path,'Ressources/betterproblem/large_objects.npy'))
large_container = np.load(os.path.join(path,'Ressources/betterproblem/large_container.npy'))
large_optimal_solution = np.load(os.path.join(path, 'Ressources/betterproblem/large_optimal.npy'))

# store in array 
objects = [small_objects, medium_objects, large_objects]
capacities = [small_container, medium_container, large_container]
runtimes = [None] * 3
for index ,obj_arr in enumerate(objects):
    number_generations = 50
    print(f'Anzahl der Objekte: {len(obj_arr)}')
    GA = GeneticAlgorithm(obj_arr, 70, capacities[index][0], capacities[index][1], 0.8, 0.05, number_generations)
    solution, all_time_best, average_vals, best_vals, worst_vals, runtimes[index] = GA.run()
    print(f'Runtime: {runtimes[index]}')
    print(f'All time best {all_time_best}')
    # Plot
    x_vals = np.arange(number_generations)
    # Punkte visualisieren
    fig, ax = plt.subplots(3, figsize=(16,7))
    ax[0].set_title(f'Average Number of bins for {len(obj_arr)}')
    ax[0].plot(x_vals, average_vals, color="blue", label = 'GeneticAlgorithm')
    ax[1].set_title('Best Number of bins')
    ax[1].plot(x_vals, best_vals, color="blue", label = 'GeneticAlgorithm')
    ax[2].set_title('Worst Number of bins')
    ax[2].plot(x_vals, worst_vals, color="blue", label = 'GeneticAlgorithm')
    plt.ylabel('Generation/Iteration')
    plt.xlabel("Number of Bins")
    plt.legend()
    plt.show()

