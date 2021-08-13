import os
import time
import numpy as np
import random
import copy


class GeneticAlgorithm:
    all_time_best = 1000000
    def __init__(self, objects, population_size, bin_vol_capacity, bin_weight_capacity,crossover_probability,mutation_probability, number_generations):
        # Create initial Population
        object_list = []
        for obj_tuple in objects:
            volume, weight = obj_tuple
            obj = Obj(volume, weight)
            object_list.append(obj)
        self.current_population = Population.create_initial_population(population_size, object_list, bin_vol_capacity, bin_weight_capacity)
        self.crossover_probability = crossover_probability
        self.mutation_probability = mutation_probability
        self.population_size = population_size
        self.number_generations = number_generations
        self.best_vals = np.zeros(number_generations)
        self.av_vals = np.zeros(number_generations)
        self.worst_vals = np.zeros(number_generations)
        self.all_time_best =1000000

    def run(self):
        start = time.time()
        for generation_number in np.arange(self.number_generations):
            #print()
            #print(f'Generation {generation_number}')
            #for chromosome in self.current_population.current_members:
            #    chromosome.print()
            # Choose Parents
            parents = Population.select_parents(self.current_population.current_members, self.population_size)
            # Create offspring through recombination of the parents
            offspring = Population.create_offspring(parents, self.crossover_probability)
            # Mutate the offspring
            for chromosome in offspring:
                chromosome.mutate(self.mutation_probability)
                # Inversion
                chromosome.inversion()
            # Replace Population
            self.current_population = Population.replace_population(offspring)
            # Print Info about the population
            self.save_info(self.current_population , generation_number)
        end = time.time()
        runtime = end - start
        return self.current_population,  self.all_time_best, self.av_vals, self.best_vals, self.worst_vals, runtime

    def save_info(self,current_population, generation_number):
        '''Saves/updates the info about the current population'''
        total_size = 0
        best_size = 1000
        worst_size = 0
        for chromosome in current_population.current_members:
            total_size += len(chromosome.group_part)
            if len(chromosome.group_part) <= best_size:
                best_size = len(chromosome.group_part)
            if len(chromosome.group_part) <= self.all_time_best:
                self.all_time_best = len(chromosome.group_part)
            if len(chromosome.group_part) >=worst_size:
                worst_size = len(chromosome.group_part)
        average_size = total_size / len(current_population.current_members)
        self.av_vals[generation_number] = average_size
        self.best_vals[generation_number] = best_size
        self.worst_vals[generation_number] = worst_size

class Population:
    def __init__(self, population_member_list):
        self.current_members = population_member_list

    def create_initial_population(population_size, object_list, bin_vol_capacity, bin_weight_capacity):
        '''Creates the initial_population by creating population_size chromosomes using the objects in object_list'''
        # create empty population
        pop = Population( np.empty(population_size, dtype=object))
        # Create chromosomes for the initial population
        for index in np.arange(len(pop.current_members)):
            random.shuffle(object_list)
            chromosome = Chromosome.create_chromosome(object_list, bin_vol_capacity, bin_weight_capacity)
            pop.current_members[index] = chromosome
        return pop

    def replace_population(offspring):
        '''Takes a list of chromosomes (offspring) and returns a new population'''
        return Population(offspring)

    def select_parents(population, number_parents):
        '''Selects number_parents from the given population using roulette_wheel sampling'''
        # TODO: Hier noch überlegen, ob man die Fitnesswerte speichert, um sie nicht immer neu zu berechnen
        # calculate the fitness for every chromosome in the population
        fitness_vals = np.empty(len(population), dtype = float)
        for index,chromosome in enumerate(population):
            fitness_vals[index] = chromosome.fitness_function()
        # create probability distribution to draw parents from
        total_fitness = np.sum(fitness_vals)
        probabilities = (1/total_fitness) * fitness_vals
        # draw number_parents with replacement
        parents = np.random.choice(population, size = len(population), replace=True, p = probabilities)
        return parents

    def create_offspring(parents, crossover_probability):
        '''Creates the offspring through recombination of the parents'''
        offpsring = []
        if len(parents) % 2 != 0:
            raise Exception("The amount of parents must not be odd.")
        else:
            # Shuffle the parents
            # TODO: Hier nochmal checken, ob wirklich nötig
            random.shuffle(parents)
            # choose two parents and recombine
            for index in np.arange(0,len(parents), 2):
                parent_a = parents[index]
                parent_b = parents[index+1]
                offspring_1, offspring_2 = Chromosome.produce_offspring(parent_a, parent_b, crossover_probability)
                offpsring = offpsring + [offspring_1, offspring_2]
        return offpsring

class Chromosome:
    def __init__(self, group_part):
        self.group_part = group_part

    def create_chromosome(object_list, bin_vol_capacity, bin_weight_capacity, prob = 0.8):
        '''Given a list of objects create a valid distribution using the first fit (chance) heuristic'''
        # create a list that only contains one bin
        group_part = [Bin(bin_vol_capacity, bin_weight_capacity)]
        # create Chromosome
        chromosome = Chromosome(group_part)
        # use first fit heuristic to distribute the objects
        for obj in object_list:
            chromosome.first_fit_chance(obj, prob)
        return chromosome

    def first_fit_chance(self, obj, prob):
        ''''With probability 1-prob fit an object obj = (Volume, Weight) into the first bin that has enough remaining capacity. With probability prob open a new bin.'''
        chance = np.random.random()
        # Use first fit
        if chance >= prob:
            for bin in self.group_part:
                if bin.check_fit(obj):
                    bin.fit_obj(obj)
                    return
            else:
                new_bin = Bin()
                self.group_part.append(new_bin)
                new_bin.fit_obj(obj)
                return
        else:
            # Open a new bin
            new_bin = Bin()
            self.group_part.append(new_bin)
            new_bin.fit_obj(obj)
            return

    def first_fit(self, obj):
        '''Fits an object obj = (Volume, Weight) into the first bin that has enough remaining capacity'''
        for bin in self.group_part:
            if bin.check_fit(obj):
                bin.fit_obj(obj)
                return
        else:
            new_bin = Bin()
            self.group_part.append(new_bin)
            new_bin.fit_obj(obj)
            return

    def random_fit(self, obj):
        #'''Fits an object obj=(Volume,Weight) into a random bin or creates a new bin.'''
        while True:
            # choose a random index
            number_bins = (len(self.group_part))
            index = np.random.randint(number_bins)+1
            # try to fit the object into a random bin
            if  index < number_bins:
                bin = self.group_part[index]
                if bin.check_fit(obj):
                    bin.fit_obj(obj)
                    return
            # create a new bin
            if index >= number_bins:
                new_bin = Bin()
                self.group_part.append(new_bin)
                new_bin.fit_obj(obj)
                return
    
    def first_random_fit(self, obj, prob = 0.3):
        # Draw a random number 
        chance = np.random.random()
        # if prob use first fit
        if chance >= prob:
            self.first_random_fit(obj)
        else:
            self.random_fit(obj)
        

    def fitness_function(self, k = 3):
        '''Calculates the fitness of the chromosome'''
        # TODO: Hier nicht mehr mit self
        # TODO: Hier noch andere Fitnessfunktionen ausprobieren
        amount_bins_used = len(self.group_part)
        numerator = 0
        for bin in self.group_part:
               numerator += ( bin.volume_fill / Bin.vol_capacity)**k+(bin.weight_fill / Bin.weight_capacity)**k
        return numerator/amount_bins_used
        #return amount_bins_used
        #return 1

    def produce_offspring(parent_chromosome_a, parent_chromosome_b, crossover_probability):
        '''Produces two offspring using the given recombination two times.'''
        offspring_1 = Chromosome.recombination(parent_chromosome_a, parent_chromosome_b, crossover_probability)
        offspring_2 = Chromosome.recombination(parent_chromosome_b, parent_chromosome_a, crossover_probability)
        return offspring_1, offspring_2

    def inversion(self):
        # TODO: Hier noch richtig implementieren
        random.shuffle(self.group_part)

    def recombination(parent_chromosome_a, parent_chromosome_b, crossover_probability, max_crossing_size = 100):
        '''Uses the BPCX to produce one offspring'''#
        # TODO: Hier vielleicht noch ein bisschen aufteilen
        # Parts of parent chromosome b are inserted into a
        # Only recombinate, if crossover_probability
        if np.random.random() <= crossover_probability:
            # choose crossing_size
            crossing_size = np.random.randint(1, max_crossing_size)
            if crossing_size > len(parent_chromosome_b.group_part):
                crossing_size = len(parent_chromosome_b.group_part)
            #print('crossing_size')
            #print(crossing_size)
            # Choose crossing point
            crossing_point = np.random.randint(0,len(parent_chromosome_b.group_part)-crossing_size+1)
            #print('Crossing Point')
            #print(crossing_point)
            # TODO: Checken ob die crossing points wirklich den gesamten Bereich abdecken (auch das Ende) Geht etwas durch den slice Operator verloren?
            bins_to_be_inserted = parent_chromosome_b.group_part[crossing_point:crossing_point + crossing_size]
            objects_to_be_inserted = []
            for bin in bins_to_be_inserted:
                objects_to_be_inserted = objects_to_be_inserted + bin.objects_contained
            # insert bins in parent_a
            # First copy the parent_chromosome_a to not change the old chromosome
            offspring_1 = parent_chromosome_a.duplicate()
            # Iterate through the bins and check which bins need to be deleted
            removed_objects = []
            for bin in reversed(offspring_1.group_part):
                # if object in bin is contained in one of the bins that were reinserted
                if set(bin.objects_contained).intersection(set(objects_to_be_inserted)):
                    # Delete the bin
                    offspring_1.group_part.remove(bin)
                    # save the objects that need to be reinserted
                    removed_objects = removed_objects + bin.objects_contained
            # choose a point where the bins will be inserted
            if len(offspring_1.group_part)==0:
                insertion_point = 0
            else:
                insertion_point= np.random.randint(0,len(offspring_1.group_part))
            # insert the bins
            #print('Insertion point')
            #print(insertion_point)
            offspring_1.group_part[insertion_point:insertion_point] = bins_to_be_inserted
            # reinsert the remaining objects using ff
            removed_objects = set(removed_objects) - set(objects_to_be_inserted)
            removed_objects = list(removed_objects)
            # sort list of removed_objects (to reinsert with first_fit (decreasing))
            # TODO: Hier noch verschiedene Sortierungen ausprobieren (chance, volume, weight .. )
            removed_objects.sort(key=lambda x: x.volume+x.weight, reverse=True)
            # reinsert using first fit
            for obj in removed_objects:
                #offspring_1.first_random_fit(obj)
                offspring_1.first_fit(obj)
                #offspring_1.random_fit(obj)
            return offspring_1
        else:
            # no recombination, offspring is identical to the parents
            offspring_1 = parent_chromosome_a.duplicate()
            return offspring_1

    #def mutate(self, mutation_probability):
    #    '''Mutate the chromosome'''
    #    # Iterate through the bins and delete bin with mutation_probability
    #    removed_objects = []
    #    for bin in reversed(self.group_part):
    #        if np.random.random() <= mutation_probability:
    #            # Delete the bin
    #            self.group_part.remove(bin)
    #            # save the objects that need to be reinserted
    #            removed_objects = removed_objects + bin.objects_contained
    #    # shuffle the objects
    #    random.shuffle(removed_objects)
    #    # use first fit to distribute the items back to the bins
    #    for obj in removed_objects:
    #        self.first_fit(obj)
    #        #self.first_random_fit(obj)
    #        #self.random_fit(obj)
#
    def mutate(self, mutation_probability):
        '''Mutates the chromosome'''
        removed_objects = []
        bins_eliminated = 0
        least_filled_bin = None
        min_filled_bin_val = 10000000
        # search the least filled bin
        for bin in self.group_part:
            if bin.weight_fill + bin.volume_fill <= min_filled_bin_val:
                min_filled_bin_val = bin.weight_fill + bin.volume_fill
                least_filled_bin = bin
        # Delete the bin
        self.group_part.remove(bin)
        bins_eliminated += 1
        # save the objects that need to be reinserted
        removed_objects = removed_objects + bin.objects_contained
        # cycle through the remaining bins and eliminate them
        for bin in reversed(self.group_part):
            if np.random.random() <= mutation_probability:
                # Delete the bin
                self.group_part.remove(bin)
                bins_eliminated += 1
                # save the objects that need to be reinserted
                removed_objects = removed_objects + bin.objects_contained
        # use first fit to distribute the items back to the bins
        # TODO: Hier noch besser aufschreiben
        # eliminate at least 3 bins 
        if bins_eliminated < 3:
            # eliminate a random bin 
            # choose random index
            bin = self.group_part[np.random.randint(0,len(self.group_part))]            
            removed_objects = removed_objects + bin.objects_contained
        if bins_eliminated < 3:
            # eliminate a random bin 
            # choose random index
            bin = self.group_part[np.random.randint(0,len(self.group_part))]            
            removed_objects = removed_objects + bin.objects_contained
        # Reinsert the objects 
        # shuffle the objects
        random.shuffle(removed_objects)
        for obj in removed_objects:
            self.first_fit(obj)
            #self.first_random_fit(obj)
            #self.random_fit(obj)

    def duplicate(self):
        '''Creates a copy of the chromosome,'''
        # no deepcopy, obj stay the same
        group_part = copy.copy(self.group_part)
        new_one = Chromosome(group_part)
        return new_one

    def print(self, only_size= False):
        print('------------------------------------------------------------------------------')
        if only_size:
            print('Amount of Bins used:' +str(len(self.group_part)))
        else:
            print('Amount of Bins used:' +str(len(self.group_part)))
            print(f'Information about the bins:')
            for bin in self.group_part:
                bin.print()
        print('------------------------------------------------------------------------------')
        print('Fitness value')
        print(self.fitness_function())
        print('------------------------------------------------------------------------------')
        print()
        print()



class Bin:
    vol_capacity = None
    weight_capacity = None

    def __init__(self, bin_vol_capacity = None, bin_weight_capacity = None):
        self.volume_fill = 0
        self.weight_fill = 0
        self.objects_contained = []
        # set the capacities if the have not been set
        if Bin.vol_capacity == None and Bin.weight_capacity == None:
            Bin.vol_capacity = bin_vol_capacity
            Bin.weight_capacity = bin_weight_capacity

    def contains_obj(self, obj):
        '''Checks if an obj is contained in a bin'''
        if obj in self.objects_contained:
            return True
        else:
            return False

    def check_fit(self, obj):
        '''Checks if the given object fits in the bin'''
        if Bin.vol_capacity - self.volume_fill >= obj.volume and Bin.weight_capacity - self.weight_fill >= obj.weight:
            return True
        else:
            return False

    def fit_obj(self, obj):
        '''Inserts an object into a bin'''
        self.volume_fill += obj.volume
        self.weight_fill += obj.weight
        self.objects_contained.append(obj)

    def print(self):
        print('Bin:')
        print(f'Capacity: volume{Bin.vol_capacity}, weight{Bin.weight_capacity}')
        print(f'Fill: volume {self.volume_fill}, weight {self.weight_fill}')
        print('Contained Objects')
        for obj in self.objects_contained:
            obj.print()

class Obj:
    def __init__(self,volume, weight):
        self.volume = volume
        self.weight = weight

    def print(self):
        print(f'Object ({self.volume},{self.weight})')
