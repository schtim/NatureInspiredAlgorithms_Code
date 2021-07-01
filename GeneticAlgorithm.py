import numpy as np

def genetic_algorithm(initial_population, generation_number,  selection_function, fitness_function, recombination_function, mutation_function,  inversion_function, replacement_function):
    population = initial_population
    for run in range(generation_number):
        # select parents from the current population
        parents = selection_function(population, fitness_function)
        # use recombination to produce offspring
        offspring = recombination_function(parents, crossover_rate)

def selection_function(population, fitness_function):
    pass

def recombination_function(parents, crossover_rate):
    pass


class Chromosome:
    def __init__(self, object_list, bin_vol_capacity, bin_weight_capacity):
        '''Given a list of objects the constructor will create a valid distribution to the bins using the first fit heuristic'''
        # the object part describes in which bin each object is located using the original index
        self.object_part = object_list
        # create a list that only contains one bin
        self.group_part = [Bin(bin_vol_capacity, bin_weight_capacity)]
        # use first fit heuristic to distribute the objects
        for obj in object_list:
            self.first_fit(obj)

    def first_fit(self, obj):
        # fits an object obj = (Volume, Weight) into the first bin that has enough capacity
        for bin in self.group_part:
            if bin.check_fit(obj):
                bin.fit_obj(obj)
                return
        else:
            new_bin = Bin()
            self.group_part.append(new_bin)
            new_bin.fit_obj(obj)
            return

    def print(self):
        print('Amount of Bins used:' +str(len(self.group_part)))
        print(f'Amount of objects: {len(self.object_part)}')
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

    chromosome = Chromosome(object_list,10,10)
    chromosome.print()
