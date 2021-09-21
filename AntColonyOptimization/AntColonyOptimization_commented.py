import sys
import time
import math
import numpy as np
from numpy.random import default_rng
rng = default_rng()


class AntColonyOptimization:
	
	def __init__(self, ant_number, iterations, b, k, p, µ, problem, container_size):
		#PARAMETER
		self.objects = problem
		self.objects_float = np.array(self.objects, dtype=float)
		self.container_size = np.array(container_size, dtype=float)
		self.n = len(self.objects)				#Anzahl Objekte
		self.µ = µ								#global best update wait counter
		self.update_g_best_wait = µ
		self.ant_number = ant_number
		self.iterations = iteration
		self.b = b 								#b: regelt Stärke der Heuristik
		self.k = float(k)						#k: regelt Fitnessfunktion
		self.p = p 								#p: Zerfallsrate
		self.p_best = 0.05						#wird für t_min genutzt
		self.t_min = ((1/(1-self.p))*(1-self.p_best**(1/float(self.n))))/((self.n/2-1)*self.p_best**(1/float(self.n)))	#Pheromonmindestwert
		self.objects_b = np.array(self.objects, dtype=float)
		self.objects_b[self.objects_b==0] = 0.9
		self.objects_b = np.float_power(self.objects_b, self.b) 	#Objekte mit b potenziert
		self.g_best = [self.n, 0, 0, 0]								#Variable für Gesamtbestwert
		#Arrays für Messwerte
		self.iteration_avg_container = np.zeros(self.iterations)
		self.iteration_avg_fitness = np.zeros(self.iterations)
		self.iteration_best = np.zeros(self.iterations)
		self.iteration_best_fitness = np.zeros(self.iterations)
		self.iteration_gbest = np.zeros(self.iterations)
		self.iteration_worst = np.zeros(self.iterations)

		#Mapping der Objekte auf Pheromonmatrixzeilen
		[self.max_weight, self.max_volume] = np.argmax(self.objects, axis=0)
		self.max_weight = self.objects[self.max_weight][0]
		self.max_volume = self.objects[self.max_volume][1]
		self.objects_mapped_index_weight = np.zeros(self.n, dtype = int)		#Soll für jedes Objekt die zugehörige Zeile in der Gewichtspheromonmatrix angeben
		self.objects_mapped_index_volume = np.zeros(self.n, dtype = int)
		self.reverse_index_weight = np.zeros(self.max_weight+1)					#Soll für jede Zeile der Gewichtspheromonmatrix das zugehörige Gewicht angeben
		self.reverse_index_volume = np.zeros(self.max_volume+1)
		self.weight_index_mem = np.ones(self.max_weight+1, dtype=int)*(-1)		#Hilfsvariable für mapping
		self.volume_index_mem = np.ones(self.max_volume+1, dtype=int)*(-1)
		self.weight_index_count = np.zeros(self.max_weight+1)					#Eintrag i entspricht hier Anzahl Objekte mit zugehörigem Gewicht von Zeile/Spalte i in Gewichtsmatrix
		self.volume_index_count = np.zeros(self.max_volume+1)
		self.weight_index_size = 0
		self.volume_index_size = 0
		self.objects_weight = np.array(self.objects[:,0])
		self.objects_volume = np.array(self.objects[:,1])
		for z in range(self.n):
			self.w = self.objects_weight[z]
			if self.weight_index_mem[self.w] == -1:
				self.weight_index_mem[self.w] = z
				self.objects_mapped_index_weight[z] = self.weight_index_size
				self.reverse_index_weight[self.weight_index_size] = self.w
				self.weight_index_count[self.weight_index_size] = 1
				self.weight_index_size += 1
			else:
				self.objects_mapped_index_weight[z] = self.objects_mapped_index_weight[self.weight_index_mem[self.w]]
				self.weight_index_count[self.objects_mapped_index_weight[z]] += 1
			self.v = self.objects_volume[z]
			if self.volume_index_mem[self.v] == -1:
				self.volume_index_mem[self.v] = z
				self.objects_mapped_index_volume[z] = self.volume_index_size
				self.reverse_index_volume[self.volume_index_size] = self.v
				self.volume_index_count[self.volume_index_size] = 1
				self.volume_index_size += 1
			else:
				self.objects_mapped_index_volume[z] = self.objects_mapped_index_volume[self.volume_index_mem[self.v]]
				self.volume_index_count[self.objects_mapped_index_volume[z]] += 1
		self.weight_index_count = self.weight_index_count[0:self.weight_index_size]
		self.volume_index_count = self.volume_index_count[0:self.volume_index_size]
		self.reverse_index_weight = self.reverse_index_weight[0:self.weight_index_size]
		self.reverse_index_volume = self.reverse_index_volume[0:self.volume_index_size]

		#Initialisierung der Pheromonmatrizen
		self.permutation_matrix_weight = np.zeros((self.n, self.weight_index_size))	#Matrix die Für jedes Objekt eine zugehörige Zeile aus der Gewichtsheromonmatrix enthalten soll
		self.permutation_matrix_volume = np.zeros((self.n, self.volume_index_size))
		for i in range(self.n):
			self.permutation_matrix_weight[i][self.objects_mapped_index_weight[i]] = 1.0
			self.permutation_matrix_volume[i][self.objects_mapped_index_volume[i]] = 1.0
		self.diag_weight = np.diag(np.diag(np.tile(self.weight_index_count, (self.weight_index_size, 1))))	#Diagonalmatrix mit Anzahl Objekte mit Gewicht i in Eintrag (i, i)
		self.diag_volume = np.diag(np.diag(np.tile(self.volume_index_count, (self.volume_index_size, 1))))
		self.pheromones_weight = np.ones((self.weight_index_size, self.weight_index_size))					#Pheromonmatrix für Gewicht
		self.pheromones_volume = np.ones((self.volume_index_size, self.volume_index_size))					#Pheromonmatrix für Volumen
		self.pheromones_weight = self.pheromones_weight*(1/(1-self.p))
		self.pheromones_volume = self.pheromones_volume*(1/(1-self.p))

	def run(self):
		start = time.time()
		for x in range(self.iterations):
			self.available_pheromones_weight = np.dot(self.permutation_matrix_weight, self.pheromones_weight) #Hole passende Zeilen der Gewichtspheromonmatrix für jedes Objekt
			self.available_pheromones_volume = np.dot(self.permutation_matrix_volume, self.pheromones_volume)
			self.solutions = ConstructAntSolutions(self.objects_float, self.objects_b, self.container_size, self.weight_index_size, self.volume_index_size, self.objects_mapped_index_weight, self.objects_mapped_index_volume, self.reverse_index_weight, self.reverse_index_volume, self.available_pheromones_weight, self.available_pheromones_volume, self.ant_number, self.n, self.k, self.b)
			#Messwerte speichern
			self.iteration_avg_container[x] = np.mean(np.array(self.solutions[:,0]), axis=None)
			self.iteration_avg_fitness[x] = np.mean(np.array(self.solutions[:,1]), axis=None)
			self.iteration_worst[x] = self.solutions[self.ant_number-1][0]
			self.iteration_best[x] = self.solutions[0][0]
			self.iteration_best_fitness[x] = self.solutions[0][1]

			#globalen Bestwert updaten
			if self.g_best[1] < self.solutions[0][1]:
				self.g_best = self.solutions[0]
				self.µ = self.update_g_best_wait
			else:
				self.µ -= 1
				if self.µ <= 1:  self.solutions[0] = self.g_best

			#Messwerte speichern
			self.iteration_gbest[x] = self.g_best[0]
			self.iteration_best_fitness[x] = self.g_best[1]

			[self.pheromones_weight, self.pheromones_volume] = UpdatePheromones(self.solutions[0], self.pheromones_weight, self.pheromones_volume, self.diag_weight, self.diag_volume, self.p, self.t_min)
		
		ende = time.time()
		runtime = ende-start
		return [runtime, self.g_best[0], self.iteration_avg_container, self.iteration_avg_fitness, self.iteration_best, self.iteration_worst, self.iteration_gbest, self.iteration_best_fitness]


def ConstructAntSolutions(objects_float, objects_b, container_size, weight_index_size, volume_index_size, objects_mapped_index_weight, objects_mapped_index_volume, reverse_index_weight, reverse_index_volume, available_pheromones_weight, available_pheromones_volume, ant_number, n, k, b):
	solutions = []
	object_list = np.arange(0, n)
	available_objects = np.zeros(n)							#Für Wahrscheinlichkeitsverteilung
	container_items_weight = np.zeros(weight_index_size) 	#Speichert Anzahl unterschiedlicher Gewichte (Analoges Mapping zur Gewichtspheromonmatrix)
	container_items_volume = np.zeros(volume_index_size)
	p_value_weight = np.zeros(n)							#Für Berechnung der Gewichtswahrscheinlichkeitsverteilung
	p_value_volume = np.zeros(n)
	p_value_weight_sum = np.zeros(n)						#Soll für jedes Objekt die Gewichtspheromonsummen speichern
	p_value_volume_sum = np.zeros(n)
	item_number = 0.0

	for ant in range(ant_number):
		ant_containers_weight = np.zeros_like(available_pheromones_weight)	#Inhalte der Containter - Gewicht (Analoges Mapping zur Gewichtspheromonmatrix)
		ant_containers_volume = np.zeros_like(available_pheromones_volume)	#Inhalte der Containter - Volumen
		ant_container_level = np.array(container_size)						#Füllstand des momentan geöffneten Containers
		current_container = 0												#momentan geöffneter Container
		remaining_objects = np.ones(n)
		for x in range (2*n):
			if item_number > 0:
				#Berechne welche Objekte noch in den momentan geöffneten Container passen und noch nicht platziert wurden
				container_fit_weight = np.subtract(np.broadcast_to(ant_container_level[0], (1, n))[0], objects_float[:, 0])
				container_fit_volume = np.subtract(np.broadcast_to(ant_container_level[1], (1, n))[0], objects_float[:, 1])
				container_fit = (container_fit_weight>=0)*(container_fit_volume>=0)*remaining_objects
				if np.any(container_fit) == 0:
					current_object = -1
				else:
					#Berechne die Auswahlwahrscheinlichkeiten für die passenden Objekte
					p_value_weight_sum = np.add(p_value_weight_sum, available_pheromones_weight[:,objects_mapped_index_weight[current_object]])
					p_value_volume_sum = np.add(p_value_volume_sum, available_pheromones_volume[:,objects_mapped_index_volume[current_object]])
					p_value_weight = np.multiply(p_value_weight_sum, container_fit)
					p_value_volume = np.multiply(p_value_volume_sum, container_fit)
					p_value_weight = np.divide(p_value_weight, item_number)
					p_value_volume = np.divide(p_value_volume, item_number)
					p_value_weight = np.multiply(p_value_weight, objects_b[:,0])
					p_value_volume = np.multiply(p_value_volume, objects_b[:,1])
					p_value_weight = np.divide(p_value_weight,np.sum(p_value_weight))
					p_value_volume = np.divide(p_value_volume,np.sum(p_value_volume))
					available_objects = np.multiply(p_value_weight, p_value_volume)
					available_objects = available_objects/available_objects.sum()
					current_object = int(rng.choice(object_list, size=None, replace=True, p=available_objects))
			else:
				if np.any(remaining_objects) == 0:
					current_object = -1
				else:
					#Berechne die Auswahlwahrscheinlichkeiten für alle verblíebenen Objekte
					p_value_weight = np.multiply(remaining_objects, objects_b[:,0])
					p_value_volume = np.multiply(remaining_objects, objects_b[:,1])
					p_value_weight = np.divide(p_value_weight,np.sum(p_value_weight))
					p_value_volume = np.divide(p_value_volume,np.sum(p_value_volume))
					available_objects = np.multiply(p_value_weight, p_value_volume)
					available_objects = available_objects/available_objects.sum()
					current_object = int(rng.choice(object_list, size=None, replace=True, p=available_objects))

			if(current_object == -1):
				#Kein Objekt passt mehr in den momentanen Container oder alle Objekte sind verteilt
				#Speichere aktuellen Container:
				ant_containers_weight[current_container] = container_items_weight
				ant_containers_volume[current_container] = container_items_volume

				container_items_weight = np.zeros(weight_index_size)
				container_items_volume = np.zeros(volume_index_size)
				p_value_weight_sum = np.zeros(n)
				p_value_volume_sum = np.zeros(n)
				ant_container_level = np.array(container_size)
				#Öffne nächsten Container
				current_container += 1

				item_number = 0.0
				#Falls alle Objekte verteilt sind hat die aktuelle Ameise ihre Lösung gefunden:
				if np.sum(remaining_objects) == 0: break
			else:
				#platziere Objekt, aktualisiere Füllstand
				ant_container_level -= objects_float[current_object]
				container_items_weight[objects_mapped_index_weight[current_object]] += 1.0
				container_items_volume[objects_mapped_index_volume[current_object]] += 1.0
				remaining_objects[current_object] = 0.0
				item_number += 1.0
		ant_containers_weight = ant_containers_weight[0:current_container, :]
		ant_containers_volume = ant_containers_volume[0:current_container, :]
		#Berechne Fitness
		fitness_weight = np.einsum('i,ji->j', reverse_index_weight, ant_containers_weight)
		fitness_volume = np.einsum('i,ji->j', reverse_index_volume, ant_containers_volume)
		fitness_weight = np.divide(fitness_weight, container_size[0])
		fitness_volume = np.divide(fitness_volume, container_size[1])
		fitness_weight = np.power(fitness_weight, k)
		fitness_volume = np.power(fitness_volume, k)
		fitness_weight = np.sum(fitness_weight)/float(current_container)
		fitness_volume = np.sum(fitness_volume)/float(current_container)
		fitness_weight = np.power(fitness_weight, k)
		fitness_volume = np.power(fitness_volume, k)
		fitness = np.add(fitness_weight, fitness_volume)/2.0
		solutions.append([current_container, fitness, ant_containers_weight, ant_containers_volume])

	#Sortiere alle Lösungen nach Fitness
	solutions = sorted(solutions, key=lambda solutions: solutions[1])
	solutions = solutions[::-1]
	solutions = np.array(solutions)
	solutions.shape = (ant_number, 4)

	return solutions

def UpdatePheromones(s_update, pheromones_weight, pheromones_volume, diag_weight, diag_volume, p, t_min):
	#Berechne co_occurence Matrizen und ziehe die Diagonalmatrizen ab
	co_occurence_weight = np.array(s_update[2])
	co_occurence_volume = np.array(s_update[3])
	co_occurence_weight = np.subtract(np.dot(co_occurence_weight.T, co_occurence_weight), diag_weight)
	co_occurence_volume = np.subtract(np.dot(co_occurence_volume.T, co_occurence_volume), diag_volume)
	#Zerfallsrate anwenden
	pheromones_weight = pheromones_weight*p
	pheromones_volume = pheromones_volume*p
	#Update addieren
	update_weight = np.add(pheromones_weight, co_occurence_weight)
	update_volume = np.add(pheromones_volume, co_occurence_volume)
	#Mindestwert überprüfen
	update_weight[update_weight<t_min] = t_min
	update_volume[update_volume<t_min] = t_min

	return [update_weight, update_volume]
