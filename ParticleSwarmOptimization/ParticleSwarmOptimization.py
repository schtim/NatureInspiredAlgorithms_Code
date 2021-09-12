## Aufruf:
# PSO = ParticleSwarmOptimization(number_particles, iterations, objects, bin_max_weight, bin_max_volume, local_coefficient, global_coefficient, chaos_coefficient,  local_coefficient_change, global_coefficient_change, chaos_coefficient_change, initiate_heuristic, unfit_heuristic, chaos_heuristic)
# gbest_object_list, gbest_used_container, runtime, gbest_bins, gbest_fitness_h, average_bins, best_bins, worst_bins, average_fitness, unfit_moves_history, chaos_moves_history, heuristic_moves_history= PSO.run()
# gbest_object_list | list of integer of length number_objects
# gbest_used_container | integer
# runtime | float
# gbest_bins | array[iterations] of integer
# gbest_fitness_h | array[iterations] of integer
# average_bins | array[iterations] of floats
# best_bins | array[iterations] of integer
# worst_bins | array[iterations] of integer
# average_fitness | array[iterations] of floats
# unfit_moves_history | array[iterations] of floats
# chaos_moves_history | array[iterations] of floats
# heuristic_moves_history | array[iterations] of floats

import os
import numpy as np
import random
import time
import copy

class ParticleSwarmOptimization:
	def __init__(self, number_particles, iterations, objects, bin_max_weight, bin_max_volume, local_coefficient, global_coefficient, chaos_coefficient,  local_coefficient_change, global_coefficient_change, chaos_coefficient_change, initiate_heuristic, unfit_heuristic, chaos_heuristic):
	#def __init__(self, objects, bin_max_weight, bin_max_volume):
		self.bin_max_weight = bin_max_weight
		self.bin_max_volume = bin_max_volume
		#self.iterations = 150
		self.iterations = iterations
		self.number_objects = objects.shape[0]
		#self.number_particles = 40
		self.number_particles = number_particles
		#self.initiate_heuristic = 'random'
		self.initiate_heuristic = initiate_heuristic
		#self.unfit_heuristic = 'first_fit'
		self.unfit_heuristic = unfit_heuristic
		#self.chaos_heuristic = 'first_fit'
		self.chaos_heuristic = chaos_heuristic
		#self.c_local = 0.4
		self.c_local = local_coefficient
		#self.c_global = 0.2
		self.c_global = global_coefficient
		#self.c_chaos = 0.2
		self.c_chaos = chaos_coefficient
		self.number_changes = self.iterations - 1
		#self.local_coefficient_change = 0.2
		self.local_coefficient_change = local_coefficient_change
		#self.global_coefficient_change = 0.4
		self.global_coefficient_change = global_coefficient_change
		#self.chaos_coefficient_change = 0.2
		self.chaos_coefficient_change = chaos_coefficient_change
		self.c_local_change = (self.local_coefficient_change - self.c_local)/self.number_changes
		self.c_global_change = (self.global_coefficient_change - self.c_global)/self.number_changes
		self.c_chaos_change = (self.chaos_coefficient_change - self.c_chaos)/self.number_changes
		if self.c_local_change == 0 and self.c_global_change == 0 and self.c_chaos_change == 0:
			self.number_changes = 0
		self.particle_list = []
		for i in range(self.number_particles):
			new_particle = Particle(objects, self.number_objects, self.bin_max_weight, self.bin_max_volume, self.initiate_heuristic)
			self.particle_list.append(new_particle)
		self.gbest_object_list = []
		for i in range(self.number_objects):
			new_gbest_object = Object(objects[i][0], objects[i][1], 0)
			self.gbest_object_list.append(new_gbest_object)
		self.gbest_used_container = self.number_objects
		self.gbest_fitness = (self.number_objects**2) * ((self.bin_max_weight**2) + (self.bin_max_volume**2))
		self.update_gbest()
		self.average_fitness = np.zeros(self.iterations+1)
		self.average_bins = np.zeros(self.iterations+1)
		self.gbest_bins = np.zeros(self.iterations+1)
		self.gbest_fitness_h = np.zeros(self.iterations+1)
		self.best_bins = np.zeros(self.iterations+1)
		self.worst_bins = np.zeros(self.iterations+1)
		self.unfit_moves_history = np.zeros(self.iterations+1)
		self.chaos_moves_history = np.zeros(self.iterations+1)
		self.heuristic_moves_history = np.zeros(self.iterations+1)
		self.update_statistic(0, 0, 0)
		
	def update_gbest(self):
		min_fitness = self.particle_list[0].fitness
		particle_number = 0
		for i in range(self.number_particles):
			if self.particle_list[i].fitness < min_fitness:
				min_fitness = self.particle_list[i].fitness
				particle_number = i
		if min_fitness < self.gbest_fitness:
			self.gbest_fitness = copy.deepcopy(self.particle_list[particle_number].fitness)
			self.gbest_used_container = copy.deepcopy(self.particle_list[particle_number].used_container)
			for i in range(self.number_objects):
				self.gbest_object_list[i].container = copy.deepcopy(self.particle_list[particle_number].object_list[i].container)
	
	def update_statistic(self, iteration, unfit_moves, chaos_moves):
		bins_count = 0
		worst_bin = 0
		worst_bin_fitness = 0
		best_bin = self.number_objects
		best_bin_fitness = (self.number_objects**2) * ((self.bin_max_weight**2) + (self.bin_max_volume**2))
		average_fitness = 0
		for i in range(self.number_particles):
			bins_count += self.particle_list[i].used_container
			average_fitness += self.particle_list[i].fitness
			if self.particle_list[i].fitness > worst_bin_fitness:
				worst_bin_fitness = copy.deepcopy(self.particle_list[i].fitness)
				worst_bin = copy.deepcopy(self.particle_list[i].used_container)
			if self.particle_list[i].fitness < best_bin_fitness:
				best_bin_fitness = copy.deepcopy(self.particle_list[i].fitness)
				best_bin = copy.deepcopy(self.particle_list[i].used_container)
		self.average_fitness[iteration] = copy.deepcopy(average_fitness/self.number_particles)
		self.average_bins[iteration] = copy.deepcopy(bins_count/self.number_particles)
		self.gbest_bins[iteration] = copy.deepcopy(self.gbest_used_container)
		self.gbest_fitness_h[iteration] = copy.deepcopy(self.gbest_fitness)
		self.best_bins[iteration] = copy.deepcopy(best_bin)
		self.worst_bins[iteration] = copy.deepcopy(worst_bin)
		self.unfit_moves_history[iteration] = copy.deepcopy(unfit_moves)
		self.chaos_moves_history[iteration] = copy.deepcopy(chaos_moves)
		self.heuristic_moves_history[iteration] = copy.deepcopy((unfit_moves + chaos_moves))
	
	def run(self):
		begin = time.perf_counter()
		if self.number_changes > 0:
			divider = self.number_changes + 1
			checks = list(range(1, divider))
		for i in range(self.iterations):
			if self.number_changes > 0:
				for j in checks:
					if int(i) == int((j*self.iterations)/divider):
						self.c_local += self.c_local_change
						self.c_global += self.c_global_change
						self.c_chaos += self.c_chaos_change
			unfit_moves = 0
			chaos_moves = 0
			for j in range(self.number_particles):
				u_moves, c_moves = self.particle_list[j].move(self.gbest_object_list, self.c_local, self.c_global, self.c_chaos, self.unfit_heuristic, self.chaos_heuristic)
				unfit_moves += u_moves
				chaos_moves += c_moves
			self.update_gbest()
			self.update_statistic(i+1, (unfit_moves/self.number_particles), (chaos_moves/self.number_particles))
		end = time.perf_counter() - begin
		runtime = '{0:0.4f}'.format(end)
		return self.gbest_object_list, self.gbest_used_container, runtime, self.gbest_bins, self.gbest_fitness_h, self.average_bins, self.best_bins, self.worst_bins, self.average_fitness, self.unfit_moves_history, self.chaos_moves_history, self.heuristic_moves_history

class Particle:
	def __init__(self, objects, number_objects, bin_max_weight, bin_max_volume, initiate_heuristic):
		self.number_objects = number_objects
		self.bin_max_weight = bin_max_weight
		self.bin_max_volume = bin_max_volume
		self.container_max = (self.bin_max_weight**2) + (self.bin_max_volume**2)
		self.initiate_heuristic = initiate_heuristic
		
		#initiate container list for particle
		self.container_list = []
		for i in range(self.number_objects):
			new_container = Container(bin_max_weight, bin_max_volume)
			self.container_list.append(new_container)
		
		#initiate object and pbest_objects list
		self.object_list = []
		self.pbest_object_list = []
		for i in range(self.number_objects):
			new_object = Object(objects[i][0], objects[i][1], 0)
			new_pbest_object = Object(objects[i][0], objects[i][1], 0)
			self.object_list.append(new_object)
			self.pbest_object_list.append(new_pbest_object)
		self.used_container = 0
		self.pbest_used_container = 0
		if initiate_heuristic == 'random':
			randomized_sequence = np.arange(0, self.number_objects).tolist()
			random.shuffle(randomized_sequence)
			for i in range(self.number_objects):
				not_moved = 1
				while(not_moved):
					new_container = random.randint(0, self.used_container)
					if self.container_list[new_container].add_object(objects[randomized_sequence[i]][0], objects[randomized_sequence[i]][1]) == 1:
						self.object_list[randomized_sequence[i]].container = new_container
						self.pbest_object_list[randomized_sequence[i]].container = new_container
						not_moved = 0
						if new_container == self.used_container:
							self.used_container += 1
							self.pbest_used_container += 1
		elif initiate_heuristic == 'random_fit':
			randomized_sequence = np.arange(0, self.number_objects).tolist()
			random.shuffle(randomized_sequence)
			max_tries = int(self.number_objects/50)
			for i in range(self.number_objects):
				not_moved = 1
				tries = 0
				while(not_moved):
					if self.used_container == 0:
						self.container_list[self.used_container].add_object(self.object_list[randomized_sequence[i]].weight, self.object_list[randomized_sequence[i]].volume)
						self.object_list[randomized_sequence[i]].container = self.used_container
						self.pbest_object_list[randomized_sequence[i]].container = self.used_container
						self.used_container += 1
						self.pbest_used_container += 1
						not_moved = 0
					elif tries > max_tries:
						if self.container_list[self.used_container].add_object(self.object_list[randomized_sequence[i]].weight, self.object_list[randomized_sequence[i]].volume) == 1:
							self.object_list[randomized_sequence[i]].container = self.used_container
							self.pbest_object_list[randomized_sequence[i]].container = self.used_container
							self.used_container += 1
							self.pbest_used_container += 1
							not_moved = 0
					else:
						new_container = random.randint(0, (self.used_container-1))
						if self.container_list[new_container].add_object(objects[randomized_sequence[i]][0], objects[randomized_sequence[i]][1]) == 1:
							self.object_list[randomized_sequence[i]].container = new_container
							self.pbest_object_list[randomized_sequence[i]].container = new_container
							not_moved = 0
						else:
							tries += 1
		elif initiate_heuristic == 'first_fit':
			randomized_sequence = np.arange(0, self.number_objects).tolist()
			random.shuffle(randomized_sequence)
			for i in range(self.number_objects):
				if self.used_container == 0:
					self.container_list[self.used_container].add_object(self.object_list[randomized_sequence[i]].weight, self.object_list[randomized_sequence[i]].volume)
					self.object_list[randomized_sequence[i]].container = self.used_container
					self.pbest_object_list[randomized_sequence[i]].container = self.used_container
					self.used_container += 1
					self.pbest_used_container += 1
				elif self.container_list[self.used_container-1].add_object(objects[randomized_sequence[i]][0], objects[randomized_sequence[i]][1]) == 1:
					self.object_list[randomized_sequence[i]].container = self.used_container-1
					self.pbest_object_list[randomized_sequence[i]].container = self.used_container-1
				else:
					if self.container_list[self.used_container].add_object(objects[randomized_sequence[i]][0], objects[randomized_sequence[i]][1]) == 1:
						self.object_list[randomized_sequence[i]].container = self.used_container
						self.pbest_object_list[randomized_sequence[i]].container = self.used_container
						self.used_container += 1
						self.pbest_used_container += 1
		self.fitness = self.get_fitness()
		self.pbest_fitness = copy.deepcopy(self.fitness)
			
	def get_fitness(self):
		punishment = (self.number_objects * self.container_max)
		temp_fitness = 0
		number_container = 0
		for i in range(self.number_objects):
			if self.container_list[i].placed_objects > 0:
				number_container += 1
				temp_fitness += (self.container_max - (self.object_list[i].weight**2) - (self.object_list[i].volume**2))
		temp_fitness += (punishment * number_container)
		return temp_fitness
	
	def get_used_container(self):
		temp_cnt = 0
		for i in range(self.number_objects):
			if self.container_list[i].placed_objects > 0:
				temp_cnt += 1
		return temp_cnt
	
	def update_pbest(self):
		if self.fitness < self.pbest_fitness:
			for i in range(self.number_objects):
				self.pbest_object_list[i].container = copy.deepcopy(self.object_list[i].container)	
			self.pbest_fitness = copy.deepcopy(self.fitness)
			self.pbest_used_container = copy.deepcopy(self.used_container)
		
	def move(self, global_object_list, local_coefficient, global_coefficient, chaos_coefficient, unfit_heuristic, chaos_heuristic):
		local_objects = []
		global_objects = []
		unfit_objects = []
		chaos_objects = []
		for i in range(self.number_objects):
			r = random.random()
			if r < local_coefficient:
				local_objects.append(i)
				self.container_list[self.object_list[i].container].remove_object(self.object_list[i].weight, self.object_list[i].volume)
			elif r < (local_coefficient + global_coefficient):
				global_objects.append(i)
				self.container_list[self.object_list[i].container].remove_object(self.object_list[i].weight, self.object_list[i].volume)
			elif r < (local_coefficient + global_coefficient + chaos_coefficient):
				chaos_objects.append(i)
				self.container_list[self.object_list[i].container].remove_object(self.object_list[i].weight, self.object_list[i].volume)
		if local_coefficient >= global_coefficient:
			random.shuffle(local_objects)
			for i in local_objects:
				if self.container_list[self.pbest_object_list[i].container].add_object(self.object_list[i].weight, self.object_list[i].volume) == 1:
					self.object_list[i].container = copy.deepcopy(self.pbest_object_list[i].container)
				else:
					unfit_objects.append(i)
			random.shuffle(global_objects)
			for i in global_objects:
				if self.container_list[global_object_list[i].container].add_object(self.object_list[i].weight, self.object_list[i].volume) == 1:
					self.object_list[i].container = copy.deepcopy(global_object_list[i].container)
				else:
					unfit_objects.append(i)
		else:
			random.shuffle(global_objects)
			for i in global_objects:
				if self.container_list[global_object_list[i].container].add_object(self.object_list[i].weight, self.object_list[i].volume) == 1:
					self.object_list[i].container = copy.deepcopy(global_object_list[i].container)
				else:
					unfit_objects.append(i)
			random.shuffle(local_objects)
			for i in local_objects:
				if self.container_list[self.pbest_object_list[i].container].add_object(self.object_list[i].weight, self.object_list[i].volume) == 1:
					self.object_list[i].container = copy.deepcopy(self.pbest_object_list[i].container)
				else:
					unfit_objects.append(i)
		unfit_moves = len(unfit_objects)
		chaos_moves = len(chaos_objects)
		if unfit_heuristic == 'random':
			random.shuffle(unfit_objects)
			filled_container = []
			empty_container = []
			for i in range(self.number_objects):
				if self.container_list[i].placed_objects > 0:
					filled_container.append(copy.deepcopy(i))
				else:
					empty_container.append(copy.deepcopy(i))
			len_fill_con = len(filled_container)
			for i in unfit_objects:
				not_moved = 1
				while(not_moved):
					new_container = random.randint(0, len_fill_con)
					if new_container == len_fill_con:
						if self.container_list[empty_container[0]].add_object(self.object_list[i].weight, self.object_list[i].volume) == 1:
							self.object_list[i].container = copy.deepcopy(empty_container[0])
							filled_container.append(empty_container[0])
							len_fill_con += 1
							empty_container.pop(0)
							not_moved = 0
					else:
						if self.container_list[filled_container[new_container]].add_object(self.object_list[i].weight, self.object_list[i].volume) == 1:
							self.object_list[i].container = copy.deepcopy(filled_container[new_container])
							not_moved = 0
		elif unfit_heuristic == 'random_fit':
			random.shuffle(unfit_objects)
			filled_container = []
			empty_container = []
			for i in range(self.number_objects):
				if self.container_list[i].placed_objects > 0:
					filled_container.append(copy.deepcopy(i))
				else:
					empty_container.append(copy.deepcopy(i))
			len_fill_con = len(filled_container)
			max_tries = 3*int(self.number_objects/50)
			for i in unfit_objects:
				not_moved = 1
				tries = 0
				while(not_moved):
					if tries > (max_tries):
						if self.container_list[empty_container[0]].add_object(self.object_list[i].weight, self.object_list[i].volume) == 1:
							self.object_list[i].container = copy.deepcopy(empty_container[0])
							filled_container.append(empty_container[0])
							len_fill_con += 1
							empty_container.pop(0)
							not_moved = 0
					else:
						new_container = random.randint(0, len_fill_con-1)
						if self.container_list[filled_container[new_container]].add_object(self.object_list[i].weight, self.object_list[i].volume) == 1:
							self.object_list[i].container = copy.deepcopy(filled_container[new_container])
							not_moved = 0
						tries += 1
		elif unfit_heuristic == 'first_fit':
			random.shuffle(unfit_objects)
			for i in unfit_objects:
				for j in range(self.number_objects):
					if self.container_list[j].add_object(self.object_list[i].weight, self.object_list[i].volume) == 1:
						self.object_list[i].container = copy.deepcopy(j)
						break
		elif unfit_heuristic == 'best_fit':
			random.shuffle(unfit_objects)
			bin_max = self.bin_max_weight + self.bin_max_volume
			for i in unfit_objects:
				best_fit_value = bin_max
				best_fit_container = 0
				empty_con = 0
				for j in range(self.number_objects):
					if self.container_list[j].placed_objects > 0 or empty_con == 0:
						temp_w = self.bin_max_weight - self.container_list[j].weight - self.object_list[i].weight
						temp_v = self.bin_max_volume - self.container_list[j].volume - self.object_list[i].volume
						if temp_w >= 0 and temp_v >= 0:
							if temp_w + temp_v < best_fit_value:
								best_fit_value = temp_w + temp_v
								best_fit_container = j
						if self.container_list[j].placed_objects == 0:
							empty_con = 1
				if self.container_list[best_fit_container].add_object(self.object_list[i].weight, self.object_list[i].volume) == 1:
						self.object_list[i].container = copy.deepcopy(best_fit_container)
		if chaos_heuristic == 'random':
			random.shuffle(chaos_objects)
			filled_container = []
			empty_container = []
			for i in range(self.number_objects):
				if self.container_list[i].placed_objects > 0:
					filled_container.append(copy.deepcopy(i))
				else:
					empty_container.append(copy.deepcopy(i))
			len_fill_con = len(filled_container)
			for i in chaos_objects:
				not_moved = 1
				while(not_moved):
					new_container = random.randint(0, len_fill_con)
					if new_container == len_fill_con:
						if self.container_list[empty_container[0]].add_object(self.object_list[i].weight, self.object_list[i].volume) == 1:
							self.object_list[i].container = copy.deepcopy(empty_container[0])
							filled_container.append(empty_container[0])
							len_fill_con += 1
							empty_container.pop(0)
							not_moved = 0
					else:
						if self.container_list[filled_container[new_container]].add_object(self.object_list[i].weight, self.object_list[i].volume) == 1:
							self.object_list[i].container = copy.deepcopy(filled_container[new_container])
							not_moved = 0
		elif chaos_heuristic == 'random_fit':
			random.shuffle(chaos_objects)
			filled_container = []
			empty_container = []
			for i in range(self.number_objects):
				if self.container_list[i].placed_objects > 0:
					filled_container.append(copy.deepcopy(i))
				else:
					empty_container.append(copy.deepcopy(i))
			len_fill_con = len(filled_container)
			max_tries = 3*int(self.number_objects/50)
			for i in chaos_objects:
				not_moved = 1
				tries = 0
				while(not_moved):
					if tries > (max_tries):
						if self.container_list[empty_container[0]].add_object(self.object_list[i].weight, self.object_list[i].volume) == 1:
							self.object_list[i].container = copy.deepcopy(empty_container[0])
							filled_container.append(empty_container[0])
							len_fill_con += 1
							empty_container.pop(0)
							not_moved = 0
					else:
						new_container = random.randint(0, len_fill_con-1)
						if self.container_list[filled_container[new_container]].add_object(self.object_list[i].weight, self.object_list[i].volume) == 1:
							self.object_list[i].container = copy.deepcopy(filled_container[new_container])
							not_moved = 0
						tries += 1
		elif chaos_heuristic == 'first_fit':
			random.shuffle(chaos_objects)
			for i in chaos_objects:
				for j in range(self.number_objects):
					if self.container_list[j].add_object(self.object_list[i].weight, self.object_list[i].volume) == 1:
						self.object_list[i].container = copy.deepcopy(j)
						break
		elif chaos_heuristic == 'best_fit':
			random.shuffle(chaos_objects)
			bin_max = self.bin_max_weight + self.bin_max_volume
			for i in chaos_objects:
				best_fit_value = bin_max
				best_fit_container = 0
				empty_con = 0
				for j in range(self.number_objects):
					if self.container_list[j].placed_objects > 0 or empty_con == 0:
						temp_w = self.bin_max_weight - self.container_list[j].weight - self.object_list[i].weight
						temp_v = self.bin_max_volume - self.container_list[j].volume - self.object_list[i].volume
						if temp_w >= 0 and temp_v >= 0:
							if temp_w + temp_v < best_fit_value:
								best_fit_value = temp_w + temp_v
								best_fit_container = j
						if self.container_list[j].placed_objects == 0:
							empty_con = 1
				if self.container_list[best_fit_container].add_object(self.object_list[i].weight, self.object_list[i].volume) == 1:
						self.object_list[i].container = copy.deepcopy(best_fit_container)
		self.used_container = copy.deepcopy(self.get_used_container())
		self.fitness = copy.deepcopy(self.get_fitness())
		self.update_pbest()
		return unfit_moves, chaos_moves
		
class Object:
	def __init__(self, object_weight, object_volume, number_container):
		self.weight = object_weight
		self.volume = object_volume
		self.container = number_container
	
	def move_to_container(self, number_container):
		self.container = number_container
		
class Container:
	def __init__(self, bin_max_weight, bin_max_volume):
		self.max_weight = bin_max_weight
		self.max_volume = bin_max_volume
		self.weight = 0
		self.volume = 0
		self.placed_objects = 0
				
	def add_object(self, object_weight, object_volume):
		if (self.weight + object_weight) <= self.max_weight and (self.volume + object_volume) <= self.max_volume:
			self.weight += object_weight
			self.volume += object_volume
			self.placed_objects += 1
			return 1
		else:
			return 0
		
	def remove_object(self, object_weight, object_volume):
		if (self.weight - object_weight) >= 0 and (self.volume - object_volume) >= 0:
			self.weight -= object_weight
			self.volume -= object_volume
			self.placed_objects -=1
			return 1
		else:
			return 0
