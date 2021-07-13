# NatureInspiredAlgorithms_Code

## Genetic Algorithm
### Fragen:
* Ich glaube, die Lösung hängt hauptsächlich von dem FirstFit ab?
* Die Auswahl der Fitnessfunktion scheint kein Einfluß auf die Lösung zu haben.
* Auch bei sehr kleinen Populationsgrößen wird die gleiche Lösung gefunden.
* Wenn nicht first_fit_chance, sondern first_fit zum initialisieren benutzt wird, ist die beste Lösung schon am Anfang erreicht 
 
### Fragen:
## Ant Colony Optimization
* Die Anzahl Ameisen und Iterationen hat wenig Einfluß auf die Qualität der Lösungen
* Stärkste Faktoren für die Qualität der Lösung sind BestFit und Strafe für leere Container bei der Containerwahl,
das sollte glaube ich nicht so sein, nach den ersten Iterationen sind oft die besten Lösungen schon gefunden, die Iterationen verbessern sich nicht kontinuierlich sondern die Verbesserungen treten eher zufällig auf, die Pheromone haben scheinbar wenig Einfluss
* Die Strafe relativ drastisch zu erhöhen/ leere Container extrem unwahrscheinlich zu machen(Faktor 1/100 - 1/1000) hat sehr starke Verbesserungen bewirkt obwohl viele Lösungen fast unerreichbar werden, Ziel sollte sein mit möglichst wenig "Strafe" auszukommen
* BestFit kann schärfer definiert werden, dadurch wurde aber keine Verbesserung erzielt
* Objekte vorher nach Gewicht oder Volumen zu sortieren hat keine Verbesserung bewirkt
## Particle Swarm Optimization
### Fragen:
* Wenn Zufallsverteilung mit FirstFit Arbeitet werden echt gute Lösungen geliefert. Blos ist es dann kein Zufall mehr und man könnte statt mit PSO auch gleich ganz mit FirstFit Arbeiten
* Aktuelle Ergebnisse noch nicht zufriefenstellend und Endergebnisse finden sich in allen Verteilungsvarianten früh in der Laufzeit (erste Hälfte) weitere Verbesserungen kosten extrem viele Schritte
* Niedrige Startcontainerzahl liefert sehr früh starke Verteilungen, die aber nur schwer weiter verbessert werden
* Nur legale Verteilungen der Objekte in den Partikeln schränkt ein, sonst würden aber illegale Verteilungen alle ihre "brauchbaren" Informationen verlieren, wie könnten gute Ergebnisse in legalen und illegalen Partikeln besser genutzt werden. Mögliches Einbauen von höherer Wahrscheinlichkeit für Objekte in gut gefüllten (legalen) Containern in diesen zu bleiben? Bin mir nicht sicher ob dies Verbesserung gibt
* Koeffizienten lieferten noch keine klaren Unterschiede, wie sie am Besten eingestellt sind, aber viel Chaos/Zufallsverteilung scheint immer gut
