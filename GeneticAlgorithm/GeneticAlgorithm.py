from os import replace
import time
import numpy as np
import random
import copy

class GeneticAlgorithm:
    all_time_best = 1000000
    number_objects = None
    def __init__(self, objects, population_size, bin_vol_capacity, bin_weight_capacity,crossover_probability,mutation_probability, number_generations, fitness_function, fit_heuristic, sampling_method = 'roulette_wheel_sampling', fit_sort = 'combined'):
        # Create initial Population
        object_list = []
        for obj_tuple in objects:
            volume, weight = obj_tuple
            obj = Obj(volume, weight)
            object_list.append(obj)
        GeneticAlgorithm.number_objects = len(object_list)
        self.current_population = Population.create_initial_population(population_size, object_list, bin_vol_capacity, bin_weight_capacity)
        self.crossover_probability = crossover_probability
        self.mutation_probability = mutation_probability
        self.population_size = population_size
        self.number_generations = number_generations
        self.best_vals = np.zeros(number_generations)
        self.av_vals = np.zeros(number_generations)
        self.worst_vals = np.zeros(number_generations)
        self.fit_heuristic = None
        self.fitness_function = None
        self.sampling_method = None
        self.all_time_best =1000000
        self.fit_sort = ''
        # set fitness function
        if fitness_function == 'amount_bins':
            self.fitness_function = Chromosome.fitness_amount_bins
        if fitness_function == 'constant':
            self.fitness_function = Chromosome.fitness_constant
        if fitness_function == 'fill':
            self.fitness_function = Chromosome.fitness_fill
        # set fit heuristic
        if fit_heuristic == 'first_fit':
            self.fit_heuristic = Chromosome.first_fit
        if fit_heuristic == 'random':
            self.fit_heuristic = Chromosome.random_fit
        # set sampling method
        if sampling_method == 'roulette_wheel_sampling':
            self.sampling_method = Population.roulette_wheel_sampling
        if sampling_method == 'tournament_selection':
            self.sampling_method = Population.tournament_selection
        if fit_sort == 'combined':
            self.fit_sort = 'combined'
        if fit_sort == 'chance':
            self.fit_sort = 'chance'
        if fit_sort =='no_sort':
            self.fit_sort = 'no_sort'
        
    def run(self):
        start = time.time()
        for generation_number in np.arange(self.number_generations):
            # Choose Parents
            parents = self.sampling_method(self.current_population.current_members, self.fitness_function)
            # Create offspring through recombination of the parents
            offspring = Population.create_offspring(parents, self.crossover_probability, self.fit_heuristic, self.fit_sort)
            # Mutate the offspring
            for chromosome in offspring:
                chromosome.mutate(self.mutation_probability, self.fit_heuristic)
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
            assert(chromosome.total_amount_objects_in_bins() == GeneticAlgorithm.number_objects), 'Chromosome with to many/ not enough objects'
            pop.current_members[index] = chromosome
        return pop

    def replace_population(offspring):
        '''Takes a list of chromosomes (offspring) and returns a new population'''
        return Population(offspring)

    def roulette_wheel_sampling(population, fitness_function):
        '''Selects number_parents from the given population using roulette_wheel sampling'''
        # TODO: Hier noch überlegen, ob man die Fitnesswerte speichert, um sie nicht immer neu zu berechnen
        # calculate the fitness for every chromosome in the population
        fitness_vals = np.empty(len(population), dtype = float)
        for index,chromosome in enumerate(population):
            fitness_vals[index] = fitness_function(chromosome)
        # create probability distribution to draw parents from
        total_fitness = np.sum(fitness_vals)
        probabilities = (1/total_fitness) * fitness_vals
        # draw number_parents with replacement
        parents = np.random.choice(population, size = len(population), replace=True, p = probabilities)
        return parents
    
    def tournament_selection(population, fitness_function, selection_pressure = 0.75):
        parents = np.empty_like(population)    
        for index in range(len(population)):
            # choose 2 chromosomes from population
            chrom_a, chrom_b = np.random.choice(population, size = 2, replace = False)
            winner, looser = Chromosome.tournament_compare(chrom_a, chrom_b, fitness_function)
            if np.random.random() <= 0.75:
                parents[index] = winner
            else:
                parents[index] = looser
        return parents


    def create_offspring(parents, crossover_probability, fit_heuristic, fit_sort):
        '''Creates the offspring through recombination of the parents'''
        offspring = []
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
                offspring_1, offspring_2 = Chromosome.produce_offspring(parent_a, parent_b, crossover_probability, fit_heuristic, fit_sort)
                offspring = offspring + [offspring_1, offspring_2]
        return offspring

class Chromosome:
    def __init__(self, group_part):
        self.group_part = group_part

    def create_chromosome(object_list, bin_vol_capacity, bin_weight_capacity, prob = 0.8):
        '''Given a list of objects create a valid distribution using the first fit (chance) heuristic'''
        # create a list that does not contain a bin yet
        group_part = [Bin.create_empty_bin(bin_vol_capacity = bin_vol_capacity, bin_weight_capacity =  bin_weight_capacity)]
        # create Chromosome
        chromosome = Chromosome(group_part)
        # use first fit heuristic to distribute the objects
        for obj in object_list:
            chromosome.first_fit_chance(obj, prob)
        #assert(chromosome.total_amount_objects_in_bins() == GeneticAlgorithm.number_objects) , 'Total number of objects is to big'
        #assert(len(chromosome.group_part) <= GeneticAlgorithm.number_objects), 'Wrong creation of chromosome'
        return chromosome

    def first_fit_chance(self, obj, prob):
        ''''With probability 1-prob fit an object obj = (Volume, Weight) into the first bin that has enough remaining capacity. With probability prob open a new bin.'''
        chance = np.random.random()
        # special case for first bin
        assert(self.object_not_contained_in_any_bin(obj)), 'Object to be inserted already contained in a bin'
        if len(self.group_part) == 1:
            if self.group_part[0].check_fit(obj):
                self.group_part[0].fit_obj(obj)
                #assert(len(self.group_part) <= GeneticAlgorithm.number_objects)
                return
            else:
                new_bin = Bin.create_empty_bin()
                self.group_part.append(new_bin)
                new_bin.fit_obj(obj)
                #assert(self.total_amount_objects_in_bins() <= GeneticAlgorithm.number_objects), 'Total number of objects to big'
                #assert(len(self.group_part) <= GeneticAlgorithm.number_objects)
                return 
        # Use first fit
        elif chance >= prob:
            for bin in self.group_part:
                if bin.check_fit(obj):
                    bin.fit_obj(obj)
                    #assert(self.total_amount_objects_in_bins() <= GeneticAlgorithm.number_objects), 'Total number of objects to big'
                    #assert(len(self.group_part) <= GeneticAlgorithm.number_objects)
                    return
                else:
                    new_bin = Bin.create_empty_bin()
                    self.group_part.append(new_bin)
                    new_bin.fit_obj(obj)
                    #assert(self.total_amount_objects_in_bins() <= GeneticAlgorithm.number_objects), 'Total number of objects to big'
                    #assert(len(self.group_part) <= GeneticAlgorithm.number_objects)
                    return
        else:
            # Open a new bin
            new_bin = Bin.create_empty_bin()
            self.group_part.append(new_bin)
            new_bin.fit_obj(obj)
            assert(self.total_amount_objects_in_bins() <= GeneticAlgorithm.number_objects), 'Total number of objects to big'
            assert(len(self.group_part) <= GeneticAlgorithm.number_objects)
            return

    def first_fit(self, obj):
        '''Fits an object obj = (Volume, Weight) into the first bin that has enough remaining capacity'''
        #assert(self.object_not_contained_in_any_bin(obj)), 'Object to be inserted already contained in a bin'
        for bin in self.group_part:
            if bin.check_fit(obj):
                bin.fit_obj(obj)
                #assert(self.total_amount_objects_in_bins() <= GeneticAlgorithm.number_objects), 'Total number of objects to big'
                #assert(len(self.group_part) <= GeneticAlgorithm.number_objects), f'{len(self.group_part)} Bins in Chromosome'
                return
        new_bin = Bin.create_empty_bin()
        self.group_part.append(new_bin)
        new_bin.fit_obj(obj)
        #assert(self.total_amount_objects_in_bins() <= GeneticAlgorithm.number_objects), 'Total number of objects to big'
        #assert(len(self.group_part) <= GeneticAlgorithm.number_objects), f'{len(self.group_part)} Bins in Chromosome'
        return

    def random_fit(self, obj):
        '''Fits an object obj=(Volume,Weight) into a random bin or creates a new bin.'''
        assert(self.object_not_contained_in_any_bin(obj)), 'Object to be inserted already contained in a bin'
        # try to fit the object 10 times 
        for _ in range(4):
            # choose a random index
            number_bins = (len(self.group_part))
            if number_bins == 0:
                new_bin = Bin.create_empty_bin()
                self.group_part.append(new_bin)
                new_bin.fit_obj(obj)
                return
            else:
                index = np.random.randint(number_bins)
            # try to fit the object into a random bin
            bin = self.group_part[index]
            if bin.check_fit(obj):
                bin.fit_obj(obj)
                #assert(self.total_amount_objects_in_bins() <= GeneticAlgorithm.number_objects), 'Total number of objects to big'
                #assert(len(self.group_part) <= GeneticAlgorithm.number_objects)
                return
            # create a new bin
        new_bin = Bin.create_empty_bin()
        self.group_part.append(new_bin)
        new_bin.fit_obj(obj)
        #assert(self.total_amount_objects_in_bins() <= GeneticAlgorithm.number_objects), 'Total number of objects to big'
        #assert(len(self.group_part) <= GeneticAlgorithm.number_objects), f'{len(self.group_part)} Bins in Chromosome'
        return
    
    def first_random_fit(self, obj, prob = 0.3):
        # Draw a random number 
        #assert(self.object_not_contained_in_any_bin(obj)), 'Object to be inserted already contained in a bin'
        chance = np.random.random()
        # if prob use first fit
        if chance >= prob:
            self.first_fit(obj)
        else:
            self.random_fit(obj)
        

    def fitness_fill(self):
        '''Calculates the fitness of the chromosome'''
        k = 2 
        amount_bins_used = len(self.group_part)
        numerator = 0
        for bin in self.group_part:
               numerator += ( bin.volume_fill / Bin.vol_capacity)**k+(bin.weight_fill / Bin.weight_capacity)**k
        return GeneticAlgorithm.number_objects +1 - amount_bins_used

    def fitness_constant(self):
        '''Calculates the fitness of the chromosome'''
        return 1
    
    def fitness_amount_bins(self):
        # TODO: Hier nochmal überlegen wie sinnvoll
        amount_bins_used = len(self.group_part)
        return GeneticAlgorithm.number_objects +1 - amount_bins_used

    def produce_offspring(parent_chromosome_a, parent_chromosome_b, crossover_probability, fit_heuristic, fit_sort):
        '''Produces two offspring using the given recombination two times.'''
        offspring_1 = Chromosome.recombination(parent_chromosome_a, parent_chromosome_b, crossover_probability, fit_heuristic, fit_sort = fit_sort)
        offspring_2 = Chromosome.recombination(parent_chromosome_b, parent_chromosome_a, crossover_probability, fit_heuristic, fit_sort = fit_sort)
        return offspring_1, offspring_2

    def inversion(self):
        random.shuffle(self.group_part)
    
    def recombination(parent_chromosome_a, parent_chromosome_b, crossover_probability, fit_heuristic, max_crossing_size = 5, fit_sort = 'combined'):
        # Parts of parent chromosome b are inserted into a
        # TODO: Hier noch einmal reinschauen
        max_crossing_size = int(0.5*len(parent_chromosome_b.group_part))+1
        # Only recombinate, if crossover_probability
        if np.random.random() <= crossover_probability:
            # choose crossing_size
            crossing_size = np.random.randint(1, max_crossing_size)
            if crossing_size > len(parent_chromosome_b.group_part):
                crossing_size = len(parent_chromosome_b.group_part)
            # Choose crossing point
            crossing_point = np.random.randint(0,len(parent_chromosome_b.group_part)-crossing_size+1)
            # TODO: Checken ob die crossing points wirklich den gesamten Bereich abdecken (auch das Ende) Geht etwas durch den slice Operator verloren?
            bins_to_be_inserted = parent_chromosome_b.group_part[crossing_point:crossing_point + crossing_size]
            # give new ids to the bins (copy them)
            temp = []
            for bin in bins_to_be_inserted:
                new_bin = bin.duplicate()
                temp.append(new_bin)
            bins_to_be_inserted = temp
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
            offspring_1.group_part[insertion_point:insertion_point] = bins_to_be_inserted
            # reinsert the remaining objects using the heuristic
            removed_objects = set(removed_objects).difference(set(objects_to_be_inserted))
            removed_objects = list(removed_objects)
            # sort list of removed_objects (to reinsert with first_fit (decreasing))
            # sort using combined 
            if fit_sort == 'combined':
                removed_objects.sort(key=lambda x: x.volume+x.weight, reverse=True)
            # sort using one of the attributes by chance 
            if fit_sort == 'chance':
                coin = np.random.random()
                if coin >= 0.5:
                    # sort using volume
                    removed_objects.sort(key=lambda x: x.volume, reverse=True)
                else:
                    # sort using one weight
                    removed_objects.sort(key=lambda x: x.weight, reverse=True)
            # reinsert using first fit
            for obj in removed_objects:
                fit_heuristic(offspring_1,obj)
            #assert(len(offspring_1.group_part)<= GeneticAlgorithm.number_objects), 'To many bins in offspring'
            #assert(offspring_1.total_amount_objects_in_bins() == GeneticAlgorithm.number_objects), f'{len(offspring_1.group_part)} Bins with to many Objects after recombination'
            #assert(parent_chromosome_a.total_amount_objects_in_bins() == GeneticAlgorithm.number_objects), f'{len(parent_chromosome_a.group_part)} Bins with to many Objects'
            #assert(parent_chromosome_b.total_amount_objects_in_bins() == GeneticAlgorithm.number_objects), f'{len(parent_chromosome_b.group_part)} Bins with to many/ not enough Objects'
            return offspring_1
        else:
            # no recombination, offspring is identical to the parents
            offspring_1 = parent_chromosome_a.duplicate()
            #assert(offspring_1.total_amount_objects_in_bins() == GeneticAlgorithm.number_objects), f'{len(offspring_1.group_part)} Bins with to many Objects after recombination'
            return offspring_1

    def mutate(self, mutation_probability, fit_heuristic):
        '''Mutate the chromosome'''
        # Iterate through the bins and delete bin with mutation_probability
        removed_objects = []
        for bin in reversed(self.group_part):
            if np.random.random() <= mutation_probability:
                # Delete the bin
                self.group_part.remove(bin)
                # save the objects that need to be reinserted
                removed_objects = removed_objects + bin.objects_contained
        # shuffle the objects
        random.shuffle(removed_objects)
        # use first fit to distribute the items back to the bins
        for obj in removed_objects:
            fit_heuristic(self,obj)
        #assert(self.total_amount_objects_in_bins() == GeneticAlgorithm.number_objects), 'Wrong number of Objects after mutation.'
        #assert(len(self.group_part) <= GeneticAlgorithm.number_objects), 'To many bins after mutation'

    #def mutate(self, mutation_probability, fit_heuristic):
    #    '''Mutates the chromosome'''
    #    removed_objects = []
    #    bins_eliminated = 0
    #    least_filled_bin = None
    #    min_filled_bin_val = 10000000
    #    # search the least filled bin
    #    for bin in self.group_part:
    #        if bin.weight_fill + bin.volume_fill <= min_filled_bin_val:
    #            min_filled_bin_val = bin.weight_fill + bin.volume_fill
    #            least_filled_bin = bin
    #    # Delete the bin
    #    self.group_part.remove(bin)
    #    bins_eliminated += 1
    #    # save the objects that need to be reinserted
    #    removed_objects = removed_objects + bin.objects_contained
    #    # cycle through the remaining bins and eliminate them
    #    for bin in reversed(self.group_part):
    #        if np.random.random() <= mutation_probability:
    #            # Delete the bin
    #            self.group_part.remove(bin)
    #            bins_eliminated += 1
    #            # save the objects that need to be reinserted
    #            removed_objects = removed_objects + bin.objects_contained
    #    # use first fit to distribute the items back to the bins
    #    # Reinsert the objects 
    #    # shuffle the objects
    #    random.shuffle(removed_objects)
    #    for obj in removed_objects:
    #        fit_heuristic(self,obj)

    def duplicate(self):
        '''Creates a copy of the chromosome,'''
        # no deepcopy, obj stay the same
        new_group_part = []
        for bin in self.group_part:
            bin_copy = bin.duplicate()
            new_group_part.append(bin_copy)
        new_one = Chromosome(new_group_part)
        return new_one

    def print(self, fitness_function, only_size= False):
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
        print(fitness_function(self))
        print('------------------------------------------------------------------------------')
        print()
        print()

    def tournament_compare(chrom_a, chrom_b, fitness_function):
        if fitness_function(chrom_a) > fitness_function(chrom_b):
            return chrom_a , chrom_b
        else:
            return chrom_b, chrom_a 

    def total_amount_objects_in_bins(self):
        '''Returns the total amount of objects in the bins'''
        sum = 0
        for bin in self.group_part:
            sum += len(bin.objects_contained)
        return sum 

    def object_not_contained_in_any_bin(self, obj):
        '''Checks if obj is contained in any of the bins '''
        for bin in self.group_part:
            if bin.contains_obj(obj):
                return False
        return True 


class Bin:
    vol_capacity = None
    weight_capacity = None

    def __init__(self, objects_contained, volume_fill, weight_fill, bin_vol_capacity = None, bin_weight_capacity = None):
        self.volume_fill = volume_fill
        self.weight_fill = weight_fill
        self.objects_contained = objects_contained
        # set the capacities if the have not been set
        if Bin.vol_capacity == None and Bin.weight_capacity == None:
            Bin.vol_capacity = bin_vol_capacity
            Bin.weight_capacity = bin_weight_capacity

    def create_empty_bin(bin_vol_capacity = None, bin_weight_capacity = None):
        ''''Creates an empty bin'''
        objects_contained = []
        return Bin(objects_contained, 0, 0, bin_vol_capacity, bin_weight_capacity)

    def duplicate(self):
        ''''Creates a shallow copy of the bin (objects stay the same)'''
        new_bin_objects_contained = copy.copy(self.objects_contained)
        new_bin = Bin(volume_fill = self.volume_fill,weight_fill= self.weight_fill, objects_contained = new_bin_objects_contained)
        return new_bin

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
        assert(self.contains_obj(obj)), 'Inserted Object that is already contained in bin'

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
        print(self)
