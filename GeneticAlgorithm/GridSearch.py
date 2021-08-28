import os
import numpy as np 
import sys
path = os.path.dirname(os.path.abspath(''))
sys.path.insert(0, path+'/GeneticAlgorithm')
from GeneticAlgorithm import GeneticAlgorithm

# implement a grid search

# define parameter vals to be searched
population_size_vals = [20,40,60,80,100]
crossover_probability_vals = [0.6,0.7,0.8,0.9]
mutation_probability_vals = [0.02,0.04,0.06,0.08,0.1,0.15,0.2]
number_generation_vals = [50,70,100,150,200]
fitness_function_vals = ['amount_bins', 'fill']

av_number = 10

# Small Problem
# load Problem

# small problem
small_objects  = np.load(os.path.join(path,'NatureInspiredAlgorithms/Compare/Ressources/betterproblem/small_objects.npy'))
small_container = np.load(os.path.join(path,'NatureInspiredAlgorithms/Compare/Ressources/betterproblem/small_container.npy'))
small_optimal_solution = np.load(os.path.join(path, 'NatureInspiredAlgorithms/Compare/Ressources/betterproblem/small_optimal.npy'))


# define output file
with open("GeneticAlgorithm/Results/Small_Grid_Res.txt", "a") as file_small:
    #perform grid search
    for pop_size in population_size_vals:
        for crossover_prob in crossover_probability_vals:
            for mutation_prob in mutation_probability_vals:
                for number_gens in number_generation_vals:
                    for fitness_function in fitness_function_vals:
                        average_vals = np.zeros((av_number, number_gens))
                        all_time_best = np.zeros(av_number)
                        runtime = np.zeros(av_number)
                        for index in np.arange(av_number):
                            GA = GeneticAlgorithm(small_objects, pop_size, small_container[0], small_container[1], crossover_prob, mutation_prob, number_gens, fitness_function, 'first_fit')
                            _,all_time_best[index], average_vals[index], _,_,runtime[index] = GA.run()
                        # average the vals
                        average_all_time_best = np.average(all_time_best)
                        global_all_time_best = np.min(all_time_best)
                        average_runtime = np.average(runtime)
                        average_average_vals = np.average(average_vals[av_number-1])
                        # write results to file
                        res_string = f'Params:\t[{pop_size:5.0f},{crossover_prob:1.2f},{mutation_prob:1.2f},{number_gens:5.0f},{fitness_function:15s}] \t Res:\t[{global_all_time_best:8.4f}{average_all_time_best:8.4f},{average_average_vals:8.4f},{average_runtime:10.2f}]'
                        file_small.write(res_string+ '\n')

print('Finished Small Problem')

## medium problem
#medium_objects  = np.load(os.path.join(path,'NatureInspiredAlgorithms/Compare/Ressources/betterproblem/medium_objects.npy'))
#medium_container = np.load(os.path.join(path,'NatureInspiredAlgorithms/Compare/Ressources/betterproblem/medium_container.npy'))
#medium_optimal_solution = np.load(os.path.join(path, 'NatureInspiredAlgorithms/Compare/Ressources/betterproblem/medium_optimal.npy'))
#
#
## define output file
#with open("GeneticAlgorithm/Results/Medium_Grid_Res.txt", "a") as file_medium:
#    #perform grid search
#    for pop_size in population_size_vals:
#        for crossover_prob in crossover_probability_vals:
#            for mutation_prob in mutation_probability_vals:
#                for number_gens in number_generation_vals:
#                    for fitness_function in fitness_function_vals:
#                        average_vals = np.zeros((av_number, number_gens))
#                        all_time_best = np.zeros(av_number)
#                        runtime = np.zeros(av_number)
#                        for index in np.arange(av_number):
#                            GA = GeneticAlgorithm(medium_objects, pop_size, medium_container[0], medium_container[1], crossover_prob, mutation_prob, number_gens, fitness_function, 'first_fit')
#                            _,all_time_best[index], average_vals[index], _,_,runtime[index] = GA.run()
#                        # average the vals
#                        average_all_time_best = np.average(all_time_best)
#                        global_all_time_best = np.min(all_time_best)
#                        average_runtime = np.average(runtime)
#                        average_average_vals = np.average(average_vals[av_number-1])
#                        # write results to file
#                        res_string = f'Params:\t[{pop_size:5.0f},{crossover_prob:1.2f},{mutation_prob:1.2f},{number_gens:5.0f},{fitness_function:15s}] \t Res:\t[{global_all_time_best:8.4f}{average_all_time_best:8.4f},{average_average_vals:8.4f},{average_runtime:10.2f}]'
#                        file_medium.write(res_string+ '\n')
## large problem
#large_objects  = np.load(os.path.join(path,'NatureInspiredAlgorithms/Compare/Ressources/betterproblem/large_objects.npy'))
#large_container = np.load(os.path.join(path,'NatureInspiredAlgorithms/Compare/Ressources/betterproblem/large_container.npy'))
#large_optimal_solution = np.load(os.path.join(path, 'NatureInspiredAlgorithms/Compare/Ressources/betterproblem/large_optimal.npy'))
#
#print('Finished Medium Problem')
#
## define output file
#with open("GeneticAlgorithm/Results/Large_Grid_Res.txt", "a") as file_large:
#    #perform grid search
#    for pop_size in population_size_vals:
#        for crossover_prob in crossover_probability_vals:
#            for mutation_prob in mutation_probability_vals:
#                for number_gens in number_generation_vals:
#                    for fitness_function in fitness_function_vals:
#                        average_vals = np.zeros((av_number, number_gens))
#                        all_time_best = np.zeros(av_number)
#                        runtime = np.zeros(av_number)
#                        for index in np.arange(av_number):
#                            GA = GeneticAlgorithm(large_objects, pop_size, large_container[0], large_container[1], crossover_prob, mutation_prob, number_gens, fitness_function, 'first_fit')
#                            _,all_time_best[index], average_vals[index], _,_,runtime[index] = GA.run()
#                        # average the vals
#                        average_all_time_best = np.average(all_time_best)
#                        global_all_time_best = np.min(all_time_best)
#                        average_runtime = np.average(runtime)
#                        average_average_vals = np.average(average_vals[av_number-1])
#                        # write results to file
#                        res_string = f'Params:\t[{pop_size:5.0f},{crossover_prob:1.2f},{mutation_prob:1.2f},{number_gens:5.0f},{fitness_function:15s}] \t Res:\t[{global_all_time_best:8.4f}{average_all_time_best:8.4f},{average_average_vals:8.4f},{average_runtime:10.2f}]'
#                        file_large.write(res_string+ '\n')