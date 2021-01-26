# Ziffern erkennen – CI

Dieses Repo enthält die Projektabgabe für das Fach "Computational Intelligence" an der Fachhochschule Erfurt. Ziel des Projektes ist es, ein neuronales Netz zu programmieren, welches in der Lage ist, die Ziffern von 0 bis 9 in kleinen Pixelrastern zu erkennen.

## Installation

Die Skripte des Projektes sind in Python geschrieben. Es muss also Python ab Version 3 installiert sein.

Zusätzlich wird die Bibliothek `numpy` verwendet, um Vektor- und Matrizenberechnungen schnell und effizient ausführen zu können. Diese kann, sofern noch nicht installiert, mittels diesem Befehl über PIP installiert werden: `pip install numpy`.

## Ausführen

Um das Hauptskript zu starten, welches die Ziffernerkennung durchführt, bitte folgenden Befehl im Terminal ausführen:

```sh
python src/main.py
```

## Projektstruktur

- `data/` enthält diverse Daten, die für die Simulation verwendet bzw. während der Simulation generiert werden.
  - `digits/` enthält die Ziffern, die es zu erkennen gilt im CSV-Format.
  - `simulations/` enthält die generierten Grafiken, die während des Ausführens des main-Skriptes erstellt werden.
- `docs/` enthält die Dokumentation – vor allem wichtig ist die Datei `projekt.md`, welche die Abgabe für das Modul beinhaltet.
- `src/` enthält alle Python-Skripte des Projektes.
  - `main.py` ist das zentrale Skript, welches die Lösung der Aufgabe enthält.
  - Alle anderen Dateien enthalten nur Hilfsfunktionen.

## Testing

In allen Dateien in `src/` sind am Ende ein paar kleine Tests geschrieben, mit denen überprüft werden kann, ob die Funktionen richtig arbeiten.

Um die Tests auszuführen, müssen die Dateien direkt mit Python als Main-Skripte ausgeführt werden. Z.B. in Bash-Shells kann das mit folgendem Befehl bewerkstelligt werden:

```sh
for f in src/*.py; do python $f; done
```
