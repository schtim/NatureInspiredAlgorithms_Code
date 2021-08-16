### PSO_values.py

import numpy as np


### ----- Anpassbare Werte -----

### Anzahl Durchläufe

iterations = 250


### Anzahl Partikel

number_particles = 20


### Containeranzahl Koeffizienten [c_start_containers <= c_max_containers]

## c_start_containers setzt die Startanzahl an Containern abhängig der Objektanzahl zum Algorithmusbeginn
c_start_containers = 0.5

### Bewegungs Koeffizienten [(c_local + c_global + c_chaos) <= 1]

## c_local beeinflusst Wahrscheinlichkeit zur Bewegung Richtung lokalem Optimum
c_local = 0.6
c_local_steps = 0.06
## c_global beeinflusst Wahrscheinlichkeit zur Bewegung Richtung globalem Optimum
c_global = 0.2
c_global_steps = 0.075
## c_chaos beeinflusst Wahrscheinlichkeit zur Bewegung zu zufälligem Container
c_chaos = 0.01
c_chaos_steps = 0.0

### Lade Container und Objekte

#container_information = np.load('Ressources/small_container.npy')
#objects = np.load('Ressources/small_objects.npy')

#container_information = np.load('Ressources/medium_container.npy')
#objects = np.load('Ressources/medium_objects.npy')

container_information = np.load('Ressources/large_container.npy')
objects = np.load('Ressources/large_objects.npy')

### ----- Feste Werte -----

### Speicher Anzahl Objekte

number_objects = int(objects.shape[0])


### Speicher Containeranzahl und Werte

start_containers = int(number_objects * c_start_containers)
max_count = int(number_objects/50)
container_max_weight = container_information[0]
container_max_volume = container_information[1]
container_max_fitness = container_max_weight**2 + container_max_volume**2
