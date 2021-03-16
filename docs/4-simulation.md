# Computersimulation

## Net-Package

Das Herzstück des Projektes ist das Net-Package, welches das neuronale Netz umsetzt. Dies ist als erstes implementiert und fertiggestellt worden.

Man kann damit neuronale Netze mit verschiedenen Neuronenzahlen auf Input-, Hidden- und Output-Schicht erzeugen. Es kann allerdings nur genau eine versteckte Schicht verwendet werden. Die Gewichte und Biaswerte werden mit Zufallswerten zwischen $-1$ und $+1$ initialisiert. Der Zufallsgenerator der Bibliothek `numpy` ist hierfür verwendet worden. Es ist möglich den Zufallsgenerator mit einem festen Startwert zu initialisieren (seeden). Dadurch wird stets die gleiche Reihe an Zufallszahlen generiert.

Die Aktivierungsfunktion ist die logistische Funktion und sie ist fest in das neuronale Netz programmiert. Gleiches bei der Fehlerfunktion, welche die Mean Squared Error Funktion ist.

Das Package ist in der Lage sowohl im Batch- als auch im Online-Modus zu arbeiten. In dieser Simulation ist allerdings nur das Batch-Verfahren verwendet worden.

## Trainings- und Testdaten

Für das Training sind fünf verschiedene Sets an Ziffern von 0 bis 9 (also insgesamt 50 Ziffern) erstellt worden. Die folgende Grafik zeigt die verschiedenen Ziffern:

![Die Beispieldatensätze – verschiedene Grafiken von Ziffern zwischen 0 und 9](img/digits.jpg)

Wie in der Aufgabe beschrieben, sind Grafiken mit 7 x 5 Pixeln erstellt worden. Ein Pixel kann entweder leer (0) oder gefüllt (1) sein.

Ziffern aus dem Set namens **"normal"** füllen das gesamte Raster aus. Sie entsprechen den typischen Darstellungen der arabischen Ziffern in geläufigen Schriftarten.

Mit dem Set **"normal-klein"** ist eine kleinere Variante des Sets "normal" geschaffen worden. Bei diesem Set bleibt die äußerste Reihe der Pixel auf allen vier Seiten frei. Vom Stil her ähneln sie aber dem Set "normal".

Ein weiteres Set trägt den Namen **"digital"**. Das komplette Raster wird von einer Ziffer ausgefüllt. Die Darstellung orientiert sich an Ziffern auf digitalen Uhren. Sie zeichnen sich durch klare gerade und einfache Linien aus.

Auch hierzu gibt es eine kleinere Variante namens **"digital-klein"**. Ähnlich wie bei "normal-klein" bleibt hier ebenfalls die äußere Pixelreihe auf allen Seiten frei. Es handelt sich um die Ziffern aus "digital" in einer kleineren Form.

Diese vier Sets bilden zusammen die *Trainingsdaten*. Sie bilden ein breites Spektrum von Ziffern in verschiedenen Darstellungen und Größen ab.

Es gibt allerdings noch ein weiteres Set namens **"evag"**. Die Erfurter Verkehrsbetriebe AG nutzt das gleiche Pixelraster von 7 x 5 Pixel auf den Anzeigetafeln an den Straßenbahn-Haltestellen in Erfurt. Die Ziffern sind denen aus dem Set "normal" sehr ähnlich. Die 0, 4 und 8 sind sogar komplett identisch. Bei allen anderen Ziffern gibt es leichte Abweichungen zu den bisherigen Sets.

Das "evag" Set stellt somit die *Testdaten*. Es wird also nicht für das Training verwendet. Stattdessen wird damit überprüft, ob das neuronale Netz auch mit unbekannten, leicht abgewandelten Datensätzen zurechtkommt.

Wie in Kapitel 3 beschrieben, sind die Datensätze in CSV-Dateien gespeichert. Sie werden durch das Digits-Module geladen und zur Verfügung gestellt.

## Ermittlung der Neuronenzahl auf der versteckten Schicht

Ab diesem Punkt findet sich die gesamte Implementierung im Hauptprogramm (siehe Kapitel 3).

In Kapitel 2 ist dargelegt worden, dass die Neuronenzahl auf der versteckten Schicht experimentell ermittelt werden muss. Dies ist also die erste Aufgabe, die es zu lösen gilt.

### Geringster initialer Fehler

Zunächst werden Netze mit 5, 10, 15, 20, 25, 30, 35, 40, 50, 60, 70, 80 und 90 Neuronen auf der versteckten Schicht betrachtet. Zu jeder Neuronenzahl werden jeweils 100 verschiedene Netze zufällig generiert (also 100 Netze mit 5 Neuronen auf der versteckten Schicht, 100 Netze mit 10 Neuronen usw.). Von diesen Netzen wird allerdings nur das Netz mit dem geringsten Fehlerwert über alle Beispieldaten ausgewählt. Am Ende bleibt also ein Sieger-Netz je Neuronenzahl übrig.

Das Ziel hierbei ist es, für jede Neuronenzahl ein Netz zu generieren, welches schon ganz gute Ausgangs-Werte liefert. Nur so können die Netze nun weiter miteinander verglichen werden.

### Fehler im Verlauf des Trainings

Als nächstes werden diese Sieger-Netze mit einer Lernrate von 0,1 über 1.000 Epochen mit den Trainingsdaten trainiert. Während des Trainings wird der Fehler über die Trainings- und Testdaten ermittelt und gespeichert. Am Ende werden die Ergebnisse miteinander verglichen.

Das Netz, welches hier die besten und schnellsten Lernerfolge zeigt und gleichzeitig gut generalisieren kann, ist der Gewinner. Dieses Netz verfügt über die geeignete Anzahl an Neuronen auf der versteckten Schicht.

## Ermittlung der optimalen Lernrate

Nachdem die optimale Anzahl an Neuronen auf der versteckten Schicht gefunden worden ist, geht es nun um die Lernrate. Auch sie ist ein wichtiger Parameter, der am ehesten experimentell ermittelt werden kann.

Es wird nun wieder die initiale Version des optimalen Netzes verwendet (d.h. die bereits durchgeführten Trainingsdurchläufe werden verworfen). Diese wird nun mehrmals mit verschiedenen Lernraten trainiert. Der Trainingsverlauf wird wieder mitverfolgt und anschließend analysiert. Es gilt die Lernrate als optimal, wo der Fehler am schnellsten, gleichmäßigsten und zuverlässigsten minimiert wird. Dauert das Training zu lange, ist sie zu klein. Wenn der Fehler an einem Punkt ein Plateau erreicht oder anfängt zu "springen", dann ist sie zu hoch.

Es werden die Lernraten 1; 0,1; 0,01 und 0,001 betrachtet. Um sichere Erkenntnisse zu gewinnen, werden 10.000 Epochen durchlaufen.

## Finales Training

Da nun (hoffentlich) die optimale Anzahl an versteckten Neuronen und eine geeignete Lernrate gefunden ist, geht es an das finale Training. Das Netz wird nun solange weitertrainiert, bis der Fehler über den Beispieldaten nahezu $0$ ist. Damit ist das Netz fertig modifiziert.

Abschließend wird mittels der Demo-App stichprobenartig analysiert, ob das Projekt tatsächlich erfolgreich gewesen ist und die Ziffern richtig erkannt werden.
