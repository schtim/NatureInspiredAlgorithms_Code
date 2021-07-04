import numpy as np
import random
import copy

class GeneticAlgorithm:
    def __init__(self):
        # TODO Hier alle Parameter eintragen und vorbereiten
        pass

    def run(self, object_list):
        pass

class Population:
    def __init__(self, population_size, object_list, bin_vol_capacity, bin_weight_capacity ):
        '''Creates the initial_population by creating population_size chromosomes using the objects in object_list'''
        self.current_members = np.empty(population_size, dtype=object)
        for index, _ in enumerate(self.current_members):
            random.shuffle(object_list)
            chromosome = Chromosome.create_chromosome(object_list, bin_vol_capacity, bin_weight_capacity)
            self.current_members[index] = chromosome

    def select_parents(population, number_parents):
        '''Selects number_parents from the given population using roulette_wheel sampling'''
        pass

    def create_offspring(parents, crossover_probability):
        '''Creates the offspring through recombination of the parents'''
        if len(parents) % 2 != 0:
            raise Exception("The amount of parents must not be odd.")
        else:
            # Shufflen der Parents
            # TODO: Hier nochmal checken, ob wirklich nötig
            random.shuffle(parents)
            for index in np.arange(0,len(parents), 2):
                print(index)



class Chromosome:
    def __init__(self, object_part, group_part):
        self.object_part = object_part
        self.group_part = group_part

    def create_chromosome(object_list, bin_vol_capacity, bin_weight_capacity):
        '''Given a list of objects the constructor will create a valid distribution to the bins using the first fit heuristic'''
        # the object part describes in which bin each object is located
        object_part = object_list
        # create a list that only contains one bin
        group_part = [Bin(bin_vol_capacity, bin_weight_capacity)]
        # create Chromosome
        chromosome = Chromosome(object_part, group_part)
        # use first fit heuristic to distribute the objects
        for obj in object_list:
            chromosome.first_fit(obj)
        return chromosome


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

    def fitness_function(self, k):
        '''Calculates the fitness of the chromosome'''
        amount_bins_used = len(self.group_part)
        numerator = 0
        for bin in self.group_part:
            numerator += ( bin.volume_fill / Bin.vol_capacity)**k+(bin.weight_fill / Bin.weight_capacity)**k
        return numerator / amount_bins_used

    def produce_offspring(parent_chromosome_a, parent_chromosome_b, crossover_probability):
        '''Produces two offspring using the given recombination two times.'''
        offspring_1 = recombination(parent_chromosome_a, parent_chromosome_b, crossover_probability)
        offspring_2 = recombination(parent_chromosome_b, parent_chromosome_a, crossover_probability)
        return offspring_1, offspring_2


    def recombination(parent_chromosome_a, parent_chromosome_b, crossover_probability):
        '''Uses the BPCX to produce one offspring'''
        # Parts of parent chromosome b are inserted into a
        # TODO: Recombination function noch implementieren
        # Only recombinate, if crossover_probability
        if np.random.random() >= crossover_probability:
            print('Recombinate')
            # Choose crossing points
            crossing_point_l = np.random.randint(0,len(parent_chromosome_b.group_part))
            crossing_point_r = np.random.randint(0,len(parent_chromosome_b.group_part))
            if crossing_point_l == crossing_point_r:
                print('Same Crossing point')
                # No Crossover takes place
                offspring_1 = parent_chromosome_a.duplicate()
                return offspring_1
            if crossing_point_l > crossing_point_r:
                # change crossing points
                cur = crossing_point_r
                crossing_point_r = crossing_point_l
                crossing_point_l = cur
            print('Crossing Points')
            print(crossing_point_l)
            print(crossing_point_r)
            # choose the bins in between the crossing_points
            # TODO: Checken ob die crossing points wirklich den gesamten Bereich abdecken (auch das Ende) Geht etwas durch den slice Operator verloren?
            bins_to_be_inserted = parent_chromosome_b.group_part[crossing_point_l:crossing_point_r]
            objects_to_be_inserted = []
            for bin in bins_to_be_inserted:
                objects_to_be_inserted = objects_to_be_inserted + bin.objects_contained
            # insert bins in parent_a
            # First copy the parent_chromosome_a to not change the old chromosome
            offspring_1 = parent_chromosome_a.duplicate()
            print('Chromosome before removing')
            offspring_1.print()
            # Iterate through the bins and check which bins need to be deleted
            removed_objects = []
            for bin in reversed(offspring_1.group_part):
                if set(bin.objects_contained).intersection(set(objects_to_be_inserted)):
                    print('HIER ein bin wurde gelöscht')
                    # Delete the bin
                    offspring_1.group_part.remove(bin)
                    # save the objects that need to be reinserted
                    removed_objects = removed_objects + bin.objects_contained
            print('Chromosome after removing')
            offspring_1.print()
            # choose a point where the bins will be inserted
            insertion_point= np.random.randint(0,len(offspring_1.group_part))
            print('Insertion point')
            print(insertion_point)
            # insert the bins
            offspring_1.group_part[insertion_point:insertion_point] = bins_to_be_inserted
            # reinsert the remaining objects using ff
            removed_objects = set(removed_objects) - set(objects_to_be_inserted)
            removed_objects = list(removed_objects)
            for obj in removed_objects:
                offspring_1.first_fit(obj)
            return offspring_1
        else:
            # no recombination, offspring is identical to the parents
            offspring_1 = parent_chromosome_a.duplicate()
            return offspring_1

    def mutate(self, mutation_probability):
        '''Mutates the chromosome'''
        # Iterate through the bins and delete bin with mutation_probability
        removed_objects = []
        for bin in reversed(self.group_part):
            if np.random.random() <= mutation_probability:
                # Delete the bin
                self.group_part.remove(bin)
                # save the objects that need to be reinserted
                removed_objects = removed_objects + bin.objects_contained
        # use first fit to distribute the items back to the bins
        for obj in removed_objects:
            self.first_fit(obj)

    def duplicate(self):
        group_part = copy.copy(self.group_part)
        object_list = copy.copy(self.object_part)
        new_one = Chromosome(object_list, group_part)
        return new_one

        return new_one

    def print(self):
        print('Amount of Bins used:' +str(len(self.group_part)))
        print(f'Amount of objects: {len(self.object_part)}')
        print(self.fitness_function(2))
        print(f'Information about the bins:')
        for bin in self.group_part:
            bin.print()

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


if __name__=='__main__':
    print()

    object_list = []
    for i in range(10):
        object_list.append(Obj(np.random.randint(1,10),np.random.randint(1,10)))

    pop = Population(100,  object_list, 10,10)

    Population.create_offspring(pop.current_members, 0.1)

    # ob = object_list[3]
    # new_list = []
    # if set(new_list).intersection(set(object_list)):
    #     print('True')

    #
    # chromosome_a = Chromosome.create_chromosome(object_list,10,10)
    # print('Chromosome A')
    # chromosome_a.print()
    # print('')
    # random.shuffle(object_list)
    # chromosome_b = Chromosome.create_chromosome(object_list,20,20)
    # print('Chromosome B')
    # chromosome_b.print()
    #
    # offspring_1 = Chromosome.produce_one_offspring(chromosome_a, chromosome_b, 0.001)
    # print('')
    # print('Offspring_1')
    # offspring_1.print()



    # obj_a = Obj(4,1)
    # obj_b = Obj(5,6)
    # obj_c = Obj(3,5)
    # obj_d = Obj(2,1)
    # obj_e = Obj(4,2)
    #
    # bin1 = Bin(10, 10)
    # bin2 = Bin(10, 10)
    #
    # bin1.fit_obj(obj_a)
    # bin1.fit_obj(obj_b)
    # bin1.fit_obj(obj_c)
    #
    # bin2.fit_obj(obj_a)
    # bin2.fit_obj(obj_d)
    # bin2.fit_obj(obj_e)
