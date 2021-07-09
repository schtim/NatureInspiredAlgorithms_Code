### Particle Swarm Optimization für das Bin Packing Problem
# Last Updated:
# 09.07.2021 19:01

# todo:
# bessere Auswertung (Durchschnittsfitness vom Partikelschwarm pro step)

import numpy
import random
import time
import copy

### Particle Swarm Optimization für das Bin Packing Problem
# Last Updated:
# 09.07.2021 19:01

# todo:
# bessere Auswertung (Durchschnittsfitness vom Partikelschwarm pro step)

import numpy
import random
import time
import copy

from PSO_values import *

class Container:
	# ein container speichert die anzahl der objekte, welche in ihm liegen, aber nicht welche objekte (wird erweitert, falls dies nötig wird)
	container_weight = 0
	container_volume = 0
	container_legal = 1
	number_objects = 0
	
	def get_weight(self):
		return self.container_weight
	
	def get_volume(self):
		return self.container_volume
	
	def get_legal(self):
		return self.container_legal
	
	def get_number_objects(self):
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
	object_position = None
	object_weight = None
	object_volume = None
	
	def __init__(self, position, weight, volume):
		self.object_position = position
		self.object_weight = weight
		self.object_volume = volume
	
	def get_position(self):
		return self.object_position
	
	def update_position(self, new_position):
		self.object_position = copy.deepcopy(new_position)
	
	def get_weight(self):
		return self.object_weight
	
	def get_volume(self):
		return self.object_volume

class Particle:
	objects_list = []
	pbest_objects_list = []
	container_list = []
	pbest_container_list = []
	particle_fitness = 0
	pbest_fitness = 0
	
	def __init__(self):
		self.objects_list = self.create_object_list()
		self.pbest_objects_list = self.create_pbest_objects_list()
		self.container_list = self.create_container_list()
		self.update_container_list()
		self.pbest_container_list = self.create_pbest_container_list()
		self.update_fitness()
		self.update_pbest()

	def get_object(self, object_number):
		return self.objects_list[object_number]

	def get_object_position(self, object_number):
		return self.get_object(object_number).get_position()
		
	def update_object_position(self, object_number, new_position):
		return self.get_object(object_number).update_position(new_position)

	def get_object_weight(self, object_number):
		return self.get_object(object_number).get_weight()
		
	def get_object_volume(self, object_number):
		return self.get_object(object_number).get_volume()
	
	def update_pbest(self):
		for i in range(number_objects):
			self.update_pbest_object_position(i, self.get_object_position(i))
		self.update_pbest_fitness()
		self.pbest_container_list = copy.deepcopy(self.container_list)
	
	def get_pbest_object(self, object_number):
		return self.pbest_objects_list[object_number]
	
	def get_pbest_object_position(self, object_number):
		return self.get_pbest_object(object_number).get_position()
		
	def update_pbest_object_position(self, object_number, new_position):
		return self.get_pbest_object(object_number).update_position(new_position)
	
	def get_container(self, container_number):
		return self.container_list[container_number]
	
	def get_container_weight(self, container_number):
		return self.get_container(container_number).get_weight()
	
	def get_container_volume(self, container_number):
		return self.get_container(container_number).get_volume()
	
	def get_container_legal(self, container_number):
		return self.get_container(container_number).get_legal()
	
	def get_container_objects(self, container_number):
		return self.get_container(container_number).get_number_objects()
	
	def inc_legal(self, container_number, weight, volume):
		return self.get_container(container_number).test_add_object(weight, volume)
	
	def inc_container(self, container_number, weight, volume):
		self.get_container(container_number).add_object(weight, volume)
	
	def dec_container(self, container_number, weight, volume):
		self.get_container(container_number).remove_object(weight, volume)

	def get_fitness(self):
		return self.particle_fitness
	
	def get_pbest_fitness(self):
		return self.pbest_fitness

	# fitnessfunktion wertet zuerst legale container mit 2 oder mehr elementen aus (hohe befüllung folgert bessere fitness)
	# dann wird die anzahl der container mit einberechnet (illegale verteilung erhält strafe: anzahl benutzter container = anzahl objekte)
	def update_fitness(self):
		legal = 1
		fitness = 0
		number_used_container = 0
		for i in range(number_objects):
			if self.get_container_legal(i) == 0:
				legal = 0
			if self.get_container_objects(i) > 0:
				number_used_container += 1
			if self.get_container_objects(i) > 1:
				if self.get_container_legal(i) == 1:
					fitness = fitness - self.get_container_weight(i) - self.get_container_volume(i)
		if legal == 1:
			fitness += self.fitness_steps(number_used_container)
		else:
			fitness += self.fitness_steps(number_objects)
		self.particle_fitness = copy.deepcopy(fitness)
	
	def fitness_steps(self, amount):
		if amount == 0:
			return 0
		else:
			return int(amount * container_max + self.fitness_steps(amount-1))
		
	def update_pbest_fitness(self):
		self.pbest_fitness = copy.deepcopy(self.get_fitness())
	
	def create_object_list(self):
		objects_list = []
		for i in range(number_objects):
			r = random.randint(0, start_containers-1)
			new_object = Object(r, objects[i][0], objects[i][1])
			objects_list.append(new_object)
		return objects_list
		
	def create_pbest_objects_list(self):
		pbest_objects_list = []
		for i in range(number_objects):
			position = self.objects_list[i].get_position()
			new_object = Object(0, objects[i][0], objects[i][1])
			pbest_objects_list.append(new_object)
		return pbest_objects_list
	
	def create_container_list(self):
		container_list = []
		for i in range(number_objects):
			container_list.append(Container())
		return container_list
	
	def create_pbest_container_list(self):
		pbest_container_list = []
		for i in range(number_objects):
			pbest_container_list.append(Container())
		return pbest_container_list
	
	def update_container_list(self):
		for i in range(number_objects):
			self.inc_container(self.get_object_position(i), self.get_object_weight(i), self.get_object_volume(i))
	
class Swarm:
	particles_list = []
	gbest_object_list = []
	gbest_particle_fitness = 0
	gbest_container_list = []
	gbest_fitness_historie = []
	gbest_container_historie = []
	
	def create_gbest(self):
		gbest_list = []
		for i in range(number_objects):
			new_object = Object(0, objects[i][0], objects[i][1])
			gbest_list.append(new_object)
		return gbest_list
	
	def create_gbest_container_list(self):
		gbest_container_list = []
		for i in range(number_objects):
			gbest_container_list.append(Container())
		return gbest_container_list
	
	def __init__(self):
		for i in range(number_particles):
			self.particles_list.append(Particle())
		self.gbest_object_list = self.create_gbest()			
		self.update_gbest(0)
			
	def get_particle(self, particel_number):
		return self.particles_list[particel_number]
	
	def get_gbest_object(self, object_number):
		return self.gbest_object_list[object_number]
	
	def get_gbest_object_position(self, object_number):
		return self.get_gbest_object(object_number).get_position()
	
	def get_gbest_object_weight(self, object_number):
		return self.get_gbest_object(object_number).get_weight()
	
	def get_gbest_object_volume(self, object_number):
		return self.get_gbest_object(object_number).get_volume()
	
	def get_gbest_fitness(self):
		return self.gbest_particle_fitness
	
	def get_gbest_number_container(self):
		number_container = 0
		for i in range(number_objects):
			if self.gbest_container_list[i].get_number_objects() > 0:
				number_container += 1
		return number_container
	
	def update_object_gbest_position(self, object_number, new_position):
		return self.get_gbest_object(object_number).update_position(new_position)

	def update_gbest_fitness(self, fitness):
		self.gbest_particle_fitness = copy.deepcopy(fitness)		

	def update_gbest(self, number_particle):
		for i in range(number_objects):
			self.update_object_gbest_position(i, self.get_particle(number_particle).get_object(i).get_position())
		self.update_gbest_fitness(self.get_particle(number_particle).get_fitness())
		self.gbest_container_list = copy.deepcopy(self.get_particle(number_particle).container_list)
		
	def add_gbest_fitness_historie(self):
		self.gbest_fitness_historie.append(copy.deepcopy(self.get_gbest_fitness()))
		
	def add_gbest_container_historie(self):
		self.gbest_container_historie.append(copy.deepcopy(self.get_gbest_number_container()))
		
	def step(self):
		for i in range(number_particles):
			objects_order = list(range(number_objects))
			random.shuffle(objects_order)
			for j in objects_order:
				not_moved = 1
				while not_moved:
					# sollte die zuweisung von r zu einem illegalem partikel führen, wird es neu generiert bis die bewegung passiert ist
					r = random.random()
					# objekt im partikel bewegt sich zu dem container in welchem es in seinem pbest befindet
					if r < c_local:
						if self.get_particle(i).inc_legal(self.get_particle(i).get_pbest_object_position(j), self.get_particle(i).get_object_weight(j), self.get_particle(i).get_object_volume(j)) == 1:
							self.get_particle(i).dec_container(self.get_particle(i).get_object_position(j), self.get_particle(i).get_object_weight(j), self.get_particle(i).get_object_volume(j))
							self.get_particle(i).update_object_position(j, self.get_particle(i).get_pbest_object_position(j))
							self.get_particle(i).inc_container(self.get_particle(i).get_pbest_object_position(j), self.get_particle(i).get_object_weight(j), self.get_particle(i).get_object_volume(j))
							not_moved = 0
						else:
							pass
					# objekt im partikel bewegt sich zu dem container in welchem es sich im gbest befindet
					elif r < (c_local+c_global):
						if self.get_particle(i).inc_legal(self.get_gbest_object_position(j), self.get_gbest_object_weight(j), self.get_gbest_object_volume(j)) == 1:
							self.get_particle(i).dec_container(self.get_particle(i).get_object_position(j), self.get_particle(i).get_object_weight(j), self.get_particle(i).get_object_volume(j))
							self.get_particle(i).update_object_position(j, self.get_gbest_object_position(j))
							self.get_particle(i).inc_container(self.get_gbest_object_position(j), self.get_gbest_object_weight(j), self.get_gbest_object_volume(j))
							not_moved = 0
						else:
							pass
					# objekt wählt seine neue position zufällig
					elif r < (c_local+c_global+c_chaos):
###----------------------------------------------------------------------------------------------------------------
### Variante Zufall in bestimmten Containern (alle container mit mind. einem element plus einen leeren container)
						jumps = []
						empty_container = 0
						for k in range(number_objects):
							if self.get_particle(i).get_container_objects(k) > 0:
								jumps.append(copy.deepcopy(k))
							elif empty_container == 0:
								jumps.append(copy.deepcopy(k))
								empty_container = 1
						jumps_len = len(jumps)
						jump_position = random.randint(0, jumps_len-1)
						while not_moved:
							if self.get_particle(i).inc_legal(jumps[jump_position], self.get_particle(i).get_object_weight(j), self.get_particle(i).get_object_volume(j)) == 1:
								self.get_particle(i).dec_container(self.get_particle(i).get_object_position(j), self.get_particle(i).get_object_weight(j), self.get_particle(i).get_object_volume(j))
								self.get_particle(i).update_object_position(j, jumps[jump_position])
								self.get_particle(i).inc_container(jumps[jump_position], self.get_particle(i).get_object_weight(j), self.get_particle(i).get_object_volume(j))
								not_moved = 0
							else:
								jump_position = random.randint(0, jumps_len-1)
###----------------------------------------------------------------------------------------------------------------
### Variante Zufall über kompletten Suchraum
#						r2 = random.randint(0, number_objects-1)
#						while not_moved:
#							if self.get_particle(i).inc_legal(r2, self.get_particle(i).get_object_weight(j), self.get_particle(i).get_object_volume(j)) == 1:
#								self.get_particle(i).dec_container(self.get_particle(i).get_object_position(j), self.get_particle(i).get_object_weight(j), self.get_particle(i).get_object_volume(j))
#								self.get_particle(i).update_object_position(j, r2)
#								self.get_particle(i).inc_container(r2, self.get_particle(i).get_object_weight(j), self.get_particle(i).get_object_volume(j))
#								not_moved = 0
#							else:
#								r2 = random.randint(0, number_objects-1)
###----------------------------------------------------------------------------------------------------------------
### Variante First Fit	(beste Ergebnisse, aber warum nicht gleich First Fit statt PSO verwenden?)				
#						k = 0
#						while not_moved:
#							if self.get_particle(i).inc_legal(k, self.get_particle(i).get_object_weight(j), self.get_particle(i).get_object_volume(j)) == 1:
#								self.get_particle(i).dec_container(self.get_particle(i).get_object_position(j), self.get_particle(i).get_object_weight(j), self.get_particle(i).get_object_volume(j))
#								self.get_particle(i).update_object_position(j, k)
#								self.get_particle(i).inc_container(k, self.get_particle(i).get_object_weight(j), self.get_particle(i).get_object_volume(j))
#								not_moved = 0
#							else:
#								k += 1
###----------------------------------------------------------------------------------------------------------------
					else:
						if self.get_particle(i).get_container_legal(j) == 1:
							not_moved = 0
		# nach bewegung: anpassen der fitness aller partikel plus updates von pbest und gbest
		for i in range(number_particles):
			self.get_particle(i).update_fitness()
			if self.get_particle(i).get_fitness() < self.get_particle(i).get_pbest_fitness():
				self.get_particle(i).update_pbest()
				if self.get_particle(i).get_fitness() < self.get_gbest_fitness():
					self.update_gbest(i)
		self.add_gbest_fitness_historie()		
		self.add_gbest_container_historie()
	
	def run(self, filename):
		f = open("Solutions/" + filename, "a")
		f.write("Anzahl Durchläufe: " + str(iterations) + "\n")
		f.write("Anzahl Partikel: " + str(number_particles) + "\n")
		f.write("Maximale Containeranzahl: " + str(number_objects) + "\n")
		self.add_gbest_fitness_historie()
		self.add_gbest_container_historie()
		for i in range(iterations):
			self.step()
		f.write("gbest Fitness Historie: " + str(self.gbest_fitness_historie) +"\n")
		f.write("gbest Container Historie: " + str(self.gbest_container_historie) +"\n")
		f.write("gbest Verteilung:\n")
		for i in range(number_objects):
			f.write(str(i) + ": " +  str(self.get_gbest_object_position(i)) + "\n")
		f.write("gbest Container: " + str(self.get_gbest_number_container()) +"\n")
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

if __name__ == "__main__":
	main()
