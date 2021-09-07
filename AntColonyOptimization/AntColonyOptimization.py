import sys
import time
import math
import numpy as np
from numpy.random import default_rng
rng = default_rng()


def AntColonyOptimization(ant_number, iterations, b, k, p, µ, s, problem):
	#ACO_BinPacking.py [ant_number] [iterations] [b:heuristic_importance] [k:fitness_stress] [p:Zerfallsrate] [µ:l_best_update] [s:solutions_that_update_pheromones][problem_size]
	#main
	objects = np.load(problem + '.npy')
	objects.shape = (int(objects.size/2), 2)
	container_size = np.load('medium_container.npy')
	container_size.shape = (2, )
	#optimals = np.load('optimal_solutions_500.npy')
	#print(optimals)
	#Parameter:
	n = len(objects)					#Anzahl Objekte
	#ant_number = int(sys.argv[1])		#Anzahl Ameisen
	#iterations = int(sys.argv[2])		#Anzahl Iterationen
	#b = float(sys.argv[3])				#Heuristik Balance (standard: 2-10)
	#k = float(sys.argv[4])				#Fitness Stress (standard: 1-2)
	#p = float(sys.argv[5])				#Zerfallsrate
	p_best = 0.05           			#approx. prob. of finding optimal solution
	t_min = ((1/(1-p))*(1-p_best**(1/float(n))))/((n/2-1)*p_best**(1/float(n)))	#lower threshold pheromones
	#µ = int(sys.argv[6])				#global best statt local best nach µ Iterationen (standard: math.ceil(500/n))
	#s = int(sys.argv[7])
	if(s > 1):
		t_min = 0
		µ = iterations
	solution_matrix = np.zeros(n*n)
	solution_matrix.shape = (n, n)
	g_best = [solution_matrix, n, 0]
	iteration_avg_container = np.zeros(iterations)
	iteration_avg_fitness = np.zeros(iterations)
	iteration_best = np.zeros(iterations)
	iteration_best_fitness = np.zeros(iterations)
	iteration_worst = np.zeros(iterations)
	iteration_worst_fitness = np.zeros(iterations)
	#initialize pheromone trails
	[max_weight, max_volume] = np.argmax(objects, axis=0)
	max_weight = objects[max_weight][0]
	max_volume = objects[max_volume][1]
	pheromones_weight = np.ones((max_weight+1, max_weight+1))
	pheromones_volume = np.ones((max_volume+1, max_volume+1))
	pheromones_weight = pheromones_weight*(1/(1-p))
	pheromones_volume = pheromones_volume*(1/(1-p))

	start = time.time()
	for x in range(iterations):
		solutions = ConstructAntSolutions(objects, container_size, max_weight, max_volume, pheromones_weight, pheromones_volume, ant_number, solution_matrix, n, k, b)
		iteration_avg_container[x] = np.mean(np.array(solutions[:,1]), axis=None)
		iteration_avg_fitness[x] = np.mean(np.array(solutions[:,2]), axis=None)
		iteration_worst[x] = solutions[ant_number-1][1]
		iteration_worst_fitness[x] = solutions[ant_number-1][2]
		iteration_best[x] = solutions[0][1]
		iteration_best_fitness[x] = solutions[0][2]
		if g_best[2] < solutions[0][2]:
			g_best = solutions[0]
			µ = int(sys.argv[6])
		else:
			µ -= 1
		[pheromones_weight, pheromones_volume] = UpdatePheromones(solutions, g_best, pheromones_weight, pheromones_volume, objects, n, p, t_min, µ, s)
	ende = time.time()
	runtime = '{:5.3f}s'.format(ende-start)
	#print(runtime)
	#print(iteration_avg_container)
	#print(iteration_avg_fitness)
	g_worst = np.argmax(iteration_worst_fitness, axis=0)
	g_worst = [iteration_worst[g_worst], iteration_worst_fitness[g_worst]]
	#print("Beste gefundene Lösung:", g_best[1])
	return [runtime, g_best[1], g_worst[0], iteration_avg_container, iteration_avg_fitness, iteration_best, iteration_best_fitness, iteration_worst, iteration_worst_fitness]
	#np.set_printoptions(threshold=np.inf)
	#print(solution[1])

def ConstructAntSolutions(objects, container_size, max_weight, max_volume, pheromones_weight, pheromones_volume, ant_number, solution_matrix, n, k, b):
	object_list = np.arange(0, n)
	objects_no_zero = np.array(objects)
	objects_no_zero[objects_no_zero==0] = 1
	container_items_weight = np.zeros(max_weight+1)
	container_items_volume = np.zeros(max_volume+1)
	item_number = 0
	solutions = []

	for ant in range(ant_number):
		ant_solution = np.array(solution_matrix)
		ant_container_level = np.array(container_size)
		current_container = 0
		remaining_objects = np.ones(n)
		fitness_weight = 0
		fitness_volume = 0

		#ants build solutions
		for x in range (2*n):
			#choose object
			available_objects = np.zeros_like(object_list)
			if item_number > 0:
				#print(np.broadcast_to(ant_container_level[0], (1, n))-objects[:, 0])
				container_fit_weight = np.broadcast_to(ant_container_level[0], (1, n))[0]-objects[:, 0]
				container_fit_volume = np.broadcast_to(ant_container_level[1], (1, n))[0]-objects[:, 1]
				#container_fit_weight = (np.tile(ant_container_level[0], (1, n)[0])-objects[:, 0])
				#container_fit_volume = (np.tile(ant_container_level[1], (1, n)[0])-objects[:, 1])
				container_fit = (container_fit_weight>=0)*(container_fit_volume>=0)*remaining_objects
				#print(container_fit)
				if np.any(container_fit) == 0:current_object = -1
				else:
					available_pheromones_weight = np.zeros((n, max_weight+1))
					available_pheromones_volume = np.zeros((n, max_volume+1))
					for x in range(n):
						if container_fit[x] == 1:
							available_pheromones_weight[x] = pheromones_weight[objects[x][0]]
							available_pheromones_volume[x] = pheromones_volume[objects[x][1]]
					p_value_weight = container_items_weight.dot(available_pheromones_weight.T)/item_number
					p_value_volume = container_items_volume.dot(available_pheromones_volume.T)/item_number
					p_value_weight = p_value_weight*objects_no_zero[:,0]**b
					p_value_volume = p_value_volume*objects_no_zero[:,1]**b
					available_objects = (p_value_weight/p_value_weight.sum(axis=None) + p_value_volume/p_value_volume.sum(axis=None))/2
					current_object = int(rng.choice(object_list, size=None, replace=True, p=available_objects))
			else:
				if np.any(remaining_objects) == 0: current_object = -1
				else:
					p_value_weight = remaining_objects*objects_no_zero[:,0]**b
					p_value_volume = remaining_objects*objects_no_zero[:,1]**b
					available_objects = (p_value_weight/p_value_weight.sum(axis=None) + p_value_volume/p_value_volume.sum(axis=None))/2
					current_object = int(rng.choice(object_list, size=None, replace=True, p=available_objects))


			if(current_object == -1):
				fitness_weight += ((container_size[0]-ant_container_level[0])/container_size[0])**k
				fitness_volume += ((container_size[1]-ant_container_level[1])/container_size[1])**k
				container_items_weight = np.zeros_like(container_items_weight)
				container_items_volume = np.zeros_like(container_items_volume)
				ant_container_level = np.array(container_size)
				current_container += 1
				item_number = 0
				if remaining_objects.sum(axis=None) == 0: break
			else:
				ant_container_level -= objects[current_object]
				ant_solution[current_object, current_container] = 1
				container_items_weight[objects[current_object][0]] += 1
				container_items_volume[objects[current_object][1]] += 1
				remaining_objects[current_object] = 0
				item_number += 1
		fitness = ((fitness_weight + fitness_volume)/current_container)*0.5
		#fitness = (fitness_weight * fitness_volume)**0.5/current_container
		solutions.append([ant_solution, current_container, fitness])
	#Sortiere Lösungen
	solutions = sorted(solutions, key=lambda solutions: solutions[2])
	solutions = solutions[::-1]
	solutions = np.array(solutions)
	solutions.shape = (ant_number, 3)

	return solutions

def UpdatePheromones(solutions, global_best, pheromones_weight, pheromones_volume, objects, n, p, t_min, µ, s):
	if µ <= 1:  solutions[0] = global_best
	update_weight = np.zeros_like(pheromones_weight)
	update_volume = np.zeros_like(pheromones_volume)
	update_sum_weight = np.zeros_like(pheromones_weight)
	update_sum_volume = np.zeros_like(pheromones_volume)
	for z in range(s):
		co_occurence_matrix = solutions[z][0].dot(solutions[z][0].T)
		np.fill_diagonal(co_occurence_matrix, 0)
		for x in range(n):
			for y in range(n-x-1):
				occurence_count = co_occurence_matrix[x][x+y+1]
				if occurence_count > 0:
					update_weight[objects[x][0]][objects[x+y+1][0]] += occurence_count
					update_volume[objects[x][1]][objects[x+y+1][1]] += occurence_count
		update_weight = update_weight + update_weight.T
		update_volume = update_volume + update_volume.T
		update_sum_weight += update_weight
		update_sum_volume += update_volume
		update_weight = np.zeros_like(pheromones_weight)
		update_volume = np.zeros_like(pheromones_volume)

	update_weight = p*pheromones_weight + update_sum_weight
	update_volume = p*pheromones_volume + update_sum_volume
	update_weight[update_weight<t_min] = t_min
	update_volume[update_volume<t_min] = t_min

	return [update_weight, update_volume]

if __name__ == "__main__":
    AntColonyOptimization(int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5]), int(sys.argv[6]), int(sys.argv[7]), sys.argv[8])