# NatureInspiredAlgorithms_Code

## Genetic Algorithm
### Fragen:
* Ich glaube, die Lösung hängt hauptsächlich von dem FirstFit ab?
* Die Auswahl der Fitnessfunktion scheint kein Einfluß auf die Lösung zu haben.
* Auch bei sehr kleinen Populationsgrößen wird die gleiche Lösung gefunden.
* Wenn nicht first_fit_chance, sondern first_fit zum initialisieren benutzt wird, ist die beste Lösung schon am Anfang erreicht 
 
### Fragen:
## Ant Colony Optimization
* Die Anzahl Ameisen und Iterationen hat wenig Einfluß auf die Qualität der Lösungen, kommt wahrscheinlich bei grösseren Eingaben mehr zum Tragen
* Stärkste Faktoren für die Qualität der Lösung sind BestFit und Strafe für leere Container bei der Containerwahl,
das sollte glaube ich nicht so sein, nach den ersten Iterationen sind oft die besten Lösungen schon gefunden, die Iterationen verbessern sich nicht kontinuierlich sondern die Verbesserungen treten scheinbar zufällig auf, die Pheromone haben scheinbar wenig Einfluss
* Die Strafe relativ drastisch zu erhöhen/ leere Container extrem unwahrscheinlich zu machen(Faktor 1/100 - 1/1000) hat sehr starke Verbesserungen bewirkt obwohl viele Lösungen fast unerreichbar werden, Ziel sollte sein mit möglichst wenig "Strafe" auszukommen
* BestFit kann schärfer definiert werden, dadurch wurde aber keine Verbesserung erzielt (momentan wird insgesamt wenig verfügbarer Platz besser bewertet, nicht wie gut ein spezielles objekt passt, zb würde ein Objekt(1,3) momentan mit gleicher Wahrscheinlichkeit in Container mit (3, 10) und (10, 3) freiem Platz zugeordnet werden)
* Objekte vorher nach Gewicht oder Volumen zu sortieren hat keine Verbesserung bewirkt
## Particle Swarm Optimization
