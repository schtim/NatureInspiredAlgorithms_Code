import numpy as np

def FindNextContainer(current_object, pheromone_trails, ant_containers, container_count, objects, n):
	container_reach = np.ones(n)
	container_reach_p = np.zeros(n)
	for x in range(n):
		if ant_containers[x][0]-objects[current_object][0]>=0 and ant_containers[x][1]-objects[current_object][1]>=0:
			container_reach[x] = x
			container_reach_p[x] = pheromone_trails[current_object, x]
			#weniger freier platz -> höhere Wertung
			container_reach_p[x] = container_reach_p[x]*1/(ant_containers[x][0]+ant_containers[x][1])
			#noch unbenutzte Container besonders schlecht bewerten
			if(container_count[x] == 0):
				container_reach_p[x] = container_reach_p[x]/10 #Ersatz für 10 finden

	p_sum = container_reach_p.sum(axis=None)
	for x in range(n):
		if container_reach_p[x]>0:
			container_reach_p[x] = container_reach_p[x]/p_sum
	current_container = np.random.choice(container_reach, size=None, replace=True, p=container_reach_p)

	return int(current_container)


def ConstructAntSolutions(objects, containers, pheromone_trails,solution_matrix, ant_count, n):
	solutions = []
	for ant in range(ant_count):
		ant_solution = np.array(solution_matrix)
		ant_containers = np.array(containers)
		container_count = np.zeros(n)
		container_count[0] = 1
		current_object = 1
		#Ameise "ant" sucht Lösung
		for x in range (n-1):
			current_container = FindNextContainer(current_object, pheromone_trails, ant_containers, container_count, objects, n)
			container_count[current_container] = 1
			#Aktualisiere Füllstand
			ant_containers[current_container, 0] -= objects[current_object, 0]
			ant_containers[current_container, 1] -= objects[current_object, 1]
			ant_solution[current_object, current_container] = 1
			current_object += 1

		print(ant_solution)
		#Eine Liste mit den Lösungen von allen Ameisen + Containerzahl
		solutions.append([int(container_count.sum(axis=None)), ant_solution])

	return solutions


def UpdatePheromones(solutions, pheromone_trails, p, l, n):
	#sortiere die Lösungen aufsteigend
	sorted_solutions = sorted(solutions, key=lambda solutions: solutions[0])

	#Die besten Lösungen verbessern ihre Kanten - tk = solution_ant_k * 1/Containeranzahl
	add_pheromones = np.array(np.zeros(n*n))
	add_pheromones.shape = (n, n)
	for x in range(l):
		add_pheromones += sorted_solutions[x][1]  * 1/sorted_solutions[x][0]

	new_pheromone_trails = (1-p)*pheromone_trails
	new_pheromone_trails += add_pheromones

	return [new_pheromone_trails, sorted_solutions[0]]


def ACObinpacking(objects, container_size):
	n = len(objects)		#Anzahl Objekte
	ant_count = n 			#Anzahl Ameisen
	p = 0.1				#Zerfallsrate
	l = 10				#Anzahl der Lösungen die Pheromonwerte erhöhen sollen

	#initzialise container array mit object_0->container_0
	containers = np.array(container_size)
	for i in range(n-1):
		containers = np.concatenate((containers, container_size))
	containers.shape = (n, 2)
	containers[0, 0] -= objects[0, 0]
	containers[0, 1] -= objects[0, 1]

	#build construction graph, initialize pheromone trails, solution shape
	pheromone_trails = np.concatenate((np.zeros(n), np.ones(n*(n-1))), axis=0)
	pheromone_trails.shape = (n, n)
	pheromone_trails[0, 0] = 1

	solution_matrix = np.zeros(n*n)
	solution_matrix.shape = (n, n)
	solution_matrix[0, 0] = 1
	best_solution = [n, solution_matrix]

	for x in range(10):	#Finde sinnvolle termination condidtion
		solutions = ConstructAntSolutions(objects, containers, pheromone_trails, solution_matrix, ant_count, n)
		update = UpdatePheromones(solutions, pheromone_trails, p, l, n)
		pheromone_trails = update[0]
		if best_solution[0]>update[1][0]:
			best_solution = update[1]

	return best_solution


#main
small_objects = np.load('small_objects.npy')
small_objects.shape = (10, 2)
small_container = np.load('small_container.npy')
small_container.shape = (2, )

medium_objects = np.load('medium_objects.npy')
medium_objects.shape = (100, 2)
medium_container = np.load('medium_container.npy')
medium_container.shape = (2,)

solution = ACObinpacking(medium_objects, medium_container)
print("Beste gefundene Lösung:")
print("Anzahl Container:", solution[0])
#np.set_printoptions(threshold=np.inf)
print(solution[1])
