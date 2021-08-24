import os
import numpy as np
import random
import time
import copy

class ParticleSwarmOptimization:
	def __init__(self, objects, number_particles, bin_max_weight, bin_max_volume, local_coefficient, global_coefficient, chaos_coefficient, number_changes, local_coefficient_change, global_coefficient_change, chaos_coefficient_change, iterations, initiate_heuristic, move_heuristic):
		self.iterations = iterations
		self.number_objects = objects.shape[0]
		self.number_particles = number_particles
		self.initiate_heuristic = initiate_heuristic
		self.move_heuristic = move_heuristic
		self.c_local = local_coefficient
		self.c_global = global_coefficient
		self.c_chaos = chaos_coefficient
		self.number_changes = number_changes
		self.c_local_change = local_coefficient_change
		self.c_global_change = global_coefficient_change
		self.c_chaos_change = chaos_coefficient_change
		self.particle_list = []
		for i in range(self.number_particles):
			new_particle = Particle(objects, self.number_objects, bin_max_weight, bin_max_volume, self.initiate_heuristic)
			self.particle_list.append(new_particle)
		self.gbest_object_list = []
		for i in range(self.number_objects):
			new_gbest_object = Object(objects[i][0], objects[i][1], 0)
			self.gbest_object_list.append(new_gbest_object)
		self.gbest_used_container = 0
		self.gbest_fitness = 100000000
		self.update_gbest()
		self.average_bins = np.zeros(self.iterations)
		self.best_bins = np.zeros(self.iterations)
		self.worst_bins = np.zeros(self.iterations)
		
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
	
	def update_statistic(self, iteration):
		bins_count = 0
		worst_bin = 0
		worst_bin_fitness = 0
		best_bin = self.number_objects
		best_bin_fitness = 100000000
		for i in range(self.number_particles):
			bins_count += self.particle_list[i].used_container
			if self.particle_list[i].fitness > worst_bin_fitness:
				worst_bin_fitness = copy.deepcopy(self.particle_list[i].fitness)
				worst_bin = copy.deepcopy(self.particle_list[i].used_container)
			if self.particle_list[i].fitness < best_bin_fitness:
				best_bin_fitness = copy.deepcopy(self.particle_list[i].fitness)
				best_bin = copy.deepcopy(self.particle_list[i].used_container)
		self.average_bins[iteration] = copy.deepcopy(bins_count/self.number_particles)
		self.best_bins[iteration] = copy.deepcopy(best_bin)
		self.worst_bins[iteration] = copy.deepcopy(worst_bin)
	
	def run(self):
		begin = time.perf_counter()
		if self.number_changes > 0:
			divider = self.number_changes + 2
			checks = list(range(2, divider))
		for i in range(self.iterations):
			if self.number_changes > 0:
				for j in checks:
					if int(i) == int((j*self.iterations)/divider):
						self.c_local += self.c_local_change
						self.c_global += self.c_global_change
						self.c_chaos += self.c_chaos_change
			for j in range(self.number_particles):
				self.particle_list[j].move(self.gbest_object_list, self.c_local, self.c_global, self.c_local, self.move_heuristic)
			self.update_gbest()
			self.update_statistic(i)
		end = time.perf_counter() - begin
		runtime = '{0:0.4f}'.format(end)
		return self.best_bins[self.iterations-1], self.average_bins, self.best_bins, self.worst_bins, runtime

class Particle:
	def __init__(self, objects, number_objects, bin_max_weight, bin_max_volume, initiate_heuristic):
		self.number_objects = number_objects
		self.bin_max_weight = bin_max_weight
		self.bin_max_volume = bin_max_volume
		self.container_max = self.bin_max_weight**2 + self.bin_max_volume**2
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
		temp_fitness = 0
		for i in range(self.number_objects):
			if self.container_list[i].placed_objects > 0:
				temp_fitness += self.container_max - self.object_list[i].weight**2 - self.object_list[i].volume**2
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
		
	def move(self, global_object_list, local_coefficient, global_coefficient, chaos_coefficient, move_heuristic):
		local_objects = []
		global_objects = []
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
					chaos_objects.append(i)
			random.shuffle(global_objects)
			for i in global_objects:
				if self.container_list[global_object_list[i].container].add_object(self.object_list[i].weight, self.object_list[i].volume) == 1:
					self.object_list[i].container = copy.deepcopy(global_object_list[i].container)
				else:
					chaos_objects.append(i)
		else:
			random.shuffle(global_objects)
			for i in global_objects:
				if self.container_list[global_object_list[i].container].add_object(self.object_list[i].weight, self.object_list[i].volume) == 1:
					self.object_list[i].container = copy.deepcopy(global_object_list[i].container)
				else:
					chaos_objects.append(i)
			random.shuffle(local_objects)
			for i in local_objects:
				if self.container_list[self.pbest_object_list[i].container].add_object(self.object_list[i].weight, self.object_list[i].volume) == 1:
					self.object_list[i].container = copy.deepcopy(self.pbest_object_list[i].container)
				else:
					chaos_objects.append(i)
		if move_heuristic == 'random':
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
		elif move_heuristic == 'random_fit':
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
		elif move_heuristic == 'first_fit':
			random.shuffle(chaos_objects)
			for i in chaos_objects:
				for j in range(self.number_objects):
					if self.container_list[j].add_object(self.object_list[i].weight, self.object_list[i].volume) == 1:
						self.object_list[i].container = copy.deepcopy(j)
						break
		elif move_heuristic == 'best_fit':
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
