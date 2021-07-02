import numpy as np
class GeneticAlgorithm:
    def __init__(self):
        # TODO Hier alle Parameter eintragen und vorbereiten
        pass

    def run(self, object_list):
        pass

class Population:
    def __init__(self, population_size, object_list, bin_vol_capacity, bin_weight_capacity ):
        '''Creates the initial_population by creating population_size chromosomes using the objects in object_list'''
        self.current_members = np.array(population_size, dtype=object)
        for index, _ in self.current_members:
            random.shuffle(object_list)
            chromosome = create_chromosome(object_list, bin_vol_capacity, bin_weight_capacity)
            self.current_members[index] = chromosome

    def select_parents(population, number_parents):
        '''Selects number_parents from the given population using roulette_wheel sampling'''
        pass

    def create_offspring(parents):
        '''Creates the offspring through recombination of the parents'''
        pass

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

    def recombination(parent_chromosome_a, parent_chromosome_b):
        '''Uses the BPCX to produce two offspring'''
        # TODO: Recombination function noch implementieren
        pass

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
        obj.corresponding_bin = self
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
        self.corresponding_bin = None

    def print(self):
        print(f'Object ({self.volume},{self.weight})')


if __name__=='__main__':
    object_list = []
    for i in range(10):
        object_list.append(Obj(np.random.randint(1,10),np.random.randint(1,10)))

    chromosome = Chromosome.create_chromosome(object_list,20,20)
    chromosome.print()
    chromosome.mutate(0.5)
    chromosome.print()
