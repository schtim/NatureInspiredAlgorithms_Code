### Particle Swarm Optimization für das Bin Packing Problem
# Last Updated:
# 15.08.2021 14:39

import numpy
import random
import time
import copy

from PSO_values import *

C_LOCAL = c_local
C_GLOBAL = c_global
C_CHAOS = c_chaos

class Container:
	# ein container speichert die anzahl der objekte, welche in ihm liegen, aber nicht welche objekte (wird erweitert, falls dies nötig wird)
	container_weight = 0
	container_volume = 0
	container_legal = 1
	number_objects = 0
	
	def get_container_weight(self):
		return self.container_weight
	
	def get_container_volume(self):
		return self.container_volume
	
	def get_container_legal(self):
		return self.container_legal
	
	def get_container_number_objects(self):
		return self.number_objects
	
	def test_add_object(self, weight, volume):
		new_weight = self.container_weight + weight
		new_volume = self.container_volume + volume
		if new_weight <= container_max_weight and new_volume <= container_max_volume:
			return 1
		else:
			return 0
	
	def add_object(self, weight, volume):
		self.container_weight += weight
		self.container_volume += volume
		if self.container_weight <= container_max_weight and self.container_volume <= container_max_volume:
			self.container_legal = 1
		else:
			self.container_legal = 0
		self.number_objects += 1
	
	def remove_object(self, weight, volume):
		self.container_weight -= weight
		self.container_volume -= volume
		if self.container_weight <= container_max_weight and self.container_volume <= container_max_volume:
			self.container_legal = 1
		else:
			self.container_legal = 0
		self.number_objects -= 1
	
class Object:
	# object_position ist die nummer des containers in welchen das objekt liegt
	number_container = None
	object_weight = None
	object_volume = None
	
	def __init__(self, container, weight, volume):
		self.number_container = container
		self.object_weight = weight
		self.object_volume = volume
	
	def get_number_container(self):
		return self.number_container
	
	def update_object_container(self, new_container):
		self.number_container = copy.deepcopy(new_container)
	
	def get_object_weight(self):
		return self.object_weight
	
	def get_object_volume(self):
		return self.object_volume

class Particle:
	objects_list = []
	container_list = []
	particle_fitness = 0
	used_container = 0
	
	#pbest
	pbest_objects_list = []
	pbest_container_list = []
	pbest_fitness = 0
	
	def __init__(self):
		self.container_list = self.create_container_list()
		self.objects_list = self.create_object_list()
		self.set_object_positions()
		self.pbest_objects_list = self.create_pbest_objects_list()
		self.pbest_container_list = self.create_pbest_container_list()
		self.update_fitness()
		self.update_used_container()
		self.update_pbest()

	# funktionen zu den objekten des partikel
	def get_object(self, object_number):
		return self.objects_list[object_number]

	def get_object_container(self, object_number):
		return self.get_object(object_number).get_number_container()
		
	def update_object_container(self, object_number, new_container):
		return self.get_object(object_number).update_object_container(new_container)

	def get_object_weight(self, object_number):
		return self.get_object(object_number).get_object_weight()
		
	def get_object_volume(self, object_number):
		return self.get_object(object_number).get_object_volume()
	
	def move_object(self, object_number, container_number):
		if self.inc_legal(container_number, self.get_object_weight(object_number), self.get_object_volume(object_number)) == 1:
			self.update_object_container(object_number, container_number)
			self.inc_container(container_number, self.get_object_weight(object_number), self.get_object_volume(object_number))
			return 1
		else:
			return 0
	
	# funktionen zu dem pbest des partikel
	def get_pbest_object(self, object_number):
		return self.pbest_objects_list[object_number]
	
	def get_pbest_object_container(self, object_number):
		return self.get_pbest_object(object_number).get_number_container()
		
	def update_pbest_object_container(self, object_number, new_container):
		return self.get_pbest_object(object_number).update_object_container(new_container)
	
	# funktionen zu den containern des partikel
	def get_container(self, container_number):
		return self.container_list[container_number]
	
	def get_container_weight(self, container_number):
		return self.get_container(container_number).get_container_weight()
	
	def get_container_volume(self, container_number):
		return self.get_container(container_number).get_container_volume()
	
	def get_container_legal(self, container_number):
		return self.get_container(container_number).get_container_legal()
	
	def get_container_objects(self, container_number):
		return self.get_container(container_number).get_container_number_objects()
	
	def inc_legal(self, container_number, weight, volume):
		return self.get_container(container_number).test_add_object(weight, volume)
	
	def inc_container(self, container_number, weight, volume):
		self.get_container(container_number).add_object(weight, volume)
	
	def dec_container(self, container_number, weight, volume):
		self.get_container(container_number).remove_object(weight, volume)

	def update_used_container(self):
		number = 0
		for i in range(number_objects):
			if self.get_container(i).get_container_number_objects() > 0:
				number+=1
		self.used_container = number
	
	def get_used_container(self):
		return self.used_container

	# funktionen zu den fitnesswerten des partikel
	def get_fitness(self):
		return self.particle_fitness
	
	def get_pbest_fitness(self):
		return self.pbest_fitness

	# fitnessfunktion wertet zuerst legale container mit 2 oder mehr elementen aus (hohe befüllung folgert bessere fitness)
	# dann wird die anzahl der container mit einberechnet (illegale verteilung erhält strafe: anzahl benutzter container = anzahl objekte)
	def update_fitness(self):
		fitness = 0
		for i in range(number_objects):
			if self.get_container_objects(i) > 0:
				fitness = fitness + container_max_fitness - self.get_container_weight(i)**2 - self.get_container_volume(i)**2
		self.particle_fitness = copy.deepcopy(fitness)
		
	# rest
	def create_object_list(self):
		o_list = []
		for i in range(number_objects):
			new_object = Object(-1, objects[i][0], objects[i][1])
			o_list.append(new_object)
		return o_list
		
	# rand-fit um initialverteilung zu verbessern, es werden number_objects/50 zufällige container getestet	
	def set_object_positions(self):	
		randomized = np.arange(0, number_objects).tolist()
		random.shuffle(randomized)
		used_con = 0
		for i in range(number_objects):
			not_moved = 1
			count = 0
			while not_moved:
				if count > max_count:
					if self.move_object(randomized[i], used_con) == 1:
						not_moved = 0
						used_con += 1
					else:
						used_con += 1
						self.move_object(randomized[i], used_con)
						not_moved = 0
				else:
					jump_position = random.randint(0, used_con)
					if self.move_object(randomized[i], jump_position) == 1:
						not_moved = 0
					else:
						count += 1
		
	def create_pbest_objects_list(self):
		pbest_objects_list = []
		for i in range(number_objects):
			position = self.objects_list[i].get_number_container()
			new_object = Object(-1, objects[i][0], objects[i][1])
			pbest_objects_list.append(new_object)
		return pbest_objects_list
	
	def create_container_list(self):
		c_list = []
		for i in range(number_objects):
			c_list.append(Container())
		return c_list
	
	def create_pbest_container_list(self):
		pbest_container_list = []
		for i in range(number_objects):
			pbest_container_list.append(Container())
		return pbest_container_list
	
	def update_pbest(self):
		for i in range(number_objects):
			self.update_pbest_object_container(i, self.get_object_container(i))
		self.pbest_fitness = copy.deepcopy(self.get_fitness())
		self.pbest_container_list = copy.deepcopy(self.container_list)
	
class Swarm:
	particles_list = []
	average_used_container = 0
		
	#gbest
	gbest_object_list = []
	gbest_particle_fitness = 0
	gbest_used_container = 0
	
	#auswertung
	gbest_fitness_historie = []
	gbest_container_historie = []
	particle_fitness_historie = []
	particle_container_historie = []
	
	def create_gbest(self):
		gbest_list = []
		for i in range(number_objects):
			new_object = Object(-1, objects[i][0], objects[i][1])
			gbest_list.append(new_object)
		return gbest_list
		
	def __init__(self):
		for i in range(number_particles):
			self.particles_list.append(Particle())
		self.gbest_object_list = self.create_gbest()			
		self.update_gbest(0)
		self.update_average_used_container()
			
	def get_particle(self, particle_number):
		return self.particles_list[particle_number]
	
	def get_average_used_container(self):
		return self.average_used_container
		
	def get_particle_used_container(self, number_particle):
		return self.get_particle(number_particle).get_used_container()
		
	def update_average_used_container(self):
		used_container = 0
		for i in range(number_particles):
			used_container += self.get_particle_used_container(i)
		self.average_used_container = used_container/number_particles
	
	def remove_particle_object_container(self, number_particle, number_object):
		self.get_particle(number_particle).dec_container(self.get_particle(number_particle).get_object_container(number_object), self.get_particle(number_particle).get_object_weight(number_object), self.get_particle(number_particle).get_object_volume(number_object))
	
	def move_particle_object(self, number_particle, number_object, new_container):
		if self.get_particle(number_particle).inc_legal(new_container, self.get_particle(number_particle).get_object_weight(number_object), self.get_particle(number_particle).get_object_volume(number_object)) == 1:
			self.get_particle(number_particle).update_object_container(number_object, new_container)
			self.get_particle(number_particle).inc_container(new_container, self.get_particle(number_particle).get_object_weight(number_object), self.get_particle(number_particle).get_object_volume(number_object))
			return 1
		else:
			return 0
	
	# gbest
	def get_gbest_object(self, object_number):
		return self.gbest_object_list[object_number]
	
	def get_gbest_object_container(self, object_number):
		return self.get_gbest_object(object_number).get_number_container()
	
	def get_gbest_fitness(self):
		return self.gbest_particle_fitness
	
	def get_gbest_used_container(self):
		return self.gbest_used_container
	
	def update_gbest_used_container(self, used_container):
		self.gbest_used_container = copy.deepcopy(used_container)
	
	def update_gbest_object_container(self, object_number, new_container):
		return self.get_gbest_object(object_number).update_object_container(new_container)
	
	def update_gbest_fitness(self, fitness):
		self.gbest_particle_fitness = copy.deepcopy(fitness)		
	
	def update_gbest(self, number_particle):
		for i in range(number_objects):
			self.update_gbest_object_container(i, self.get_particle(number_particle).get_object(i).get_number_container())
		self.update_gbest_fitness(self.get_particle(number_particle).get_fitness())
		self.update_gbest_used_container(self.get_particle(number_particle).get_used_container())
	
	# historien
	def add_gbest_fitness_historie(self):
		self.gbest_fitness_historie.append(copy.deepcopy(self.get_gbest_fitness()))
	
	def add_gbest_container_historie(self):
		self.gbest_container_historie.append(copy.deepcopy(self.get_gbest_used_container()))
	
	def add_particle_fitness_historie(self):
		fitness = 0
		for i in range(number_particles):
			fitness += self.get_particle(i).get_fitness()
		self.particle_fitness_historie.append(int(fitness/number_particles))
	
	def add_particle_container_historie(self):
		self.particle_container_historie.append(copy.deepcopy(self.get_average_used_container()))
	
	def step(self):
		for i in range(number_particles):
			objects_move_local = []
			objects_move_global = []
			objects_move_chaos = []
			for j in range(number_objects):
				r = random.random()
				if r < C_LOCAL:
					objects_move_local.append(j)
					self.remove_particle_object_container(i, j)
				elif r < (C_LOCAL+C_GLOBAL):
					objects_move_global.append(j)
					self.remove_particle_object_container(i, j)
				elif r < (C_LOCAL+C_GLOBAL+C_CHAOS):
					objects_move_chaos.append(j)
					self.remove_particle_object_container(i, j)
				else:
					pass
			if C_LOCAL > C_GLOBAL:
				random.shuffle(objects_move_local)
				for j in objects_move_local:
					if self.move_particle_object(i, j, self.get_particle(i).get_pbest_object_container(j)) == 0:
						objects_move_chaos.append(j)
				random.shuffle(objects_move_global)
				for j in objects_move_global:
					if self.move_particle_object(i, j, self.get_gbest_object_container(j)) == 0:
						objects_move_chaos.append(j)
			else:
				random.shuffle(objects_move_global)
				for j in objects_move_global:
					if self.move_particle_object(i, j, self.get_gbest_object_container(j)) == 0:
						objects_move_chaos.append(j)
				random.shuffle(objects_move_local)
				for j in objects_move_local:
					if self.move_particle_object(i, j, self.get_particle(i).get_pbest_object_container(j)) == 0:
						objects_move_chaos.append(j)
			random.shuffle(objects_move_chaos)
			con = []
			empty_con = []
			for k in range(number_objects):
				if self.get_particle(i).get_container_objects(k) > 0:
					con.append(copy.deepcopy(k))
				else:
					empty_con.append(copy.deepcopy(k))
			con_len = int(len(con))
			for j in objects_move_chaos:
				not_moved = 1
				count = 0
				while(not_moved):
					jump_position = random.randint(0, con_len-1)
					if count > max_count*3:
						if self.move_particle_object(i, j, empty_con[0]) == 1:
							con.append(copy.deepcopy(empty_con[0]))
							con_len += 1
							empty_con.pop(0)
							not_moved = 0
					elif self.move_particle_object(i, j, con[jump_position]) == 1:
						not_moved = 0
					count += 1
			# nach bewegung: anpassen der fitness aller partikel plus updates von pbest und gbest
			self.get_particle(i).update_used_container()
		for i in range(number_particles):
			self.get_particle(i).update_fitness()
			if self.get_particle(i).get_fitness() < self.get_particle(i).get_pbest_fitness():
				self.get_particle(i).update_pbest()
			if self.get_particle(i).get_fitness() < self.get_gbest_fitness():
				self.update_gbest(i)
						
	def run(self, filename):
		f = open("Solutions/" + filename, "a")
		f.write("Anzahl Durchläufe: " + str(iterations) + "\n")
		f.write("Anzahl Partikel: " + str(number_particles) + "\n")
		f.write("Koeffizienten: c_local = " + str(c_local) + " ,c_global = " + str(c_global) + " ,c_chaos = " + str(c_chaos) + "\n") 
		self.add_gbest_fitness_historie()
		self.add_gbest_container_historie()
		self.add_particle_fitness_historie()
		self.add_particle_container_historie()
		
		global C_LOCAL
		global C_GLOBAL
		global C_CHAOS
		
		for i in range(iterations):
			if int(i) == int((2*iterations)/9):
				C_LOCAL = C_LOCAL - c_local_steps
				C_GLOBAL = C_GLOBAL + c_global_steps
				C_CHAOS = C_CHAOS + c_chaos_steps
			elif int(i) == int((3*iterations)/9):
				C_LOCAL = C_LOCAL - c_local_steps
				C_GLOBAL = C_GLOBAL + c_global_steps
				C_CHAOS = C_CHAOS + c_chaos_steps
			elif int(i) == int((4*iterations)/9):
				C_LOCAL = C_LOCAL - c_local_steps
				C_GLOBAL = C_GLOBAL + c_global_steps
				C_CHAOS = C_CHAOS + c_chaos_steps
			elif int(i) == int((5*iterations)/9):
				C_LOCAL = C_LOCAL - c_local_steps
				C_GLOBAL = C_GLOBAL + c_global_steps
				C_CHAOS = C_CHAOS + c_chaos_steps
			elif int(i) == int((6*iterations)/9):
				C_LOCAL = C_LOCAL - c_local_steps
				C_GLOBAL = C_GLOBAL + c_global_steps
				C_CHAOS = C_CHAOS + c_chaos_steps
			elif int(i) == int((7*iterations)/9):
				C_LOCAL = C_LOCAL - c_local_steps
				C_GLOBAL = C_GLOBAL + c_global_steps
				C_CHAOS = C_CHAOS + c_chaos_steps
			elif int(i) == int((8*iterations)/9):
				C_LOCAL = C_LOCAL - c_local_steps
				C_GLOBAL = C_GLOBAL + c_global_steps
				C_CHAOS = C_CHAOS + c_chaos_steps
			self.step()
			self.update_average_used_container()
			#print(str(i) + " : " + str(self.average_used_container) + " : " + str(self.gbest_used_container))
			self.add_gbest_fitness_historie()		
			self.add_gbest_container_historie()
			self.add_particle_fitness_historie()
			self.add_particle_container_historie()
		f.write("gbest Container: " + str(self.get_gbest_used_container()) +"\n")
		f.write("gbest Container Historie: " + str(self.gbest_container_historie) +"\n")
		f.write("gbest Fitness Historie: " + str(self.gbest_fitness_historie) +"\n")
		f.write("Particle Container Historie: " + str(self.particle_container_historie) +"\n")
		f.write("Particle Fitness Historie: " + str(self.particle_fitness_historie) +"\n")
		f.close()		

def main():
	np.set_printoptions(threshold=np.inf)
	filename = time.strftime("%Y%m%d-%H%M%S")
	begin = time.perf_counter()
	particle_swarm = Swarm()
	particle_swarm.run(filename)
	end = time.perf_counter() - begin
	f = open("Solutions/" + filename, "a")
	f.write("Laufzeit: " + '{0:0.4f}'.format(end) + "s\n")
	f.close()
	print("Laufzeit: " + '{0:0.4f}'.format(end) + "s")

if __name__ == "__main__":
	main()
