# Umsetzung in Python

Dieses Projekt befindet sich in einem Git-Repository. Dort befinden sich alle Dateien des Projektes, inklusive Source-Code, Dokumentation usw. Das Repository ist unter diesem Link erreichbar:

<https://github.com/hd-code/digit-recognition-ci/>

## Projektübersicht

Damit das Projekt übersichtlich bleibt, ist es in Ordner und Unterordner gegliedert. Die Ordner sind:

- `data`
  - `digits` – Hier liegen die Ziffern für die Trainingsdaten als CSV-Dateien.
  - `simulations` – Hier werden die Ergebnisse einer Simulation gespeichert.
- `docs` – Hier befindet sich die Dokumentation des Projektes (also faktisch dieses Dokument)
- `src` – Hier befindet sich der Source-Code. Weitere Erklärungen dazu folgen in anschließenden Abschnitten

## Benötigte Software und Installation

Das Projekt ist in Python geschrieben. Es muss also **Python** ab *Version 3.9 oder höher* installiert sein, um die Skripte auszuführen.

Zusätzlich sollte **Pipenv** installiert sein. Pipenv kümmert sich um die Installation aller weiteren benötigten Software-Komponenten. Zusammen mit Python kommt der Package-Manager **Pip**. Mittels Pip kann Pipenv sehr einfach installiert werden, dazu muss über die Kommandozeile folgendes ausgeführt werden:

```sh
pip install pipenv
```

Nun ist Pipenv installiert. Damit können nun alle weiteren Abhängigkeiten installiert werden. Dazu bitte in der Kommandozeile in den Ordner des Projektes wechseln und diesen Befehl ausführen:

```sh
pipenv install
```

Dadurch werden die weiteren benötigten Software-Komponenten installiert. Diese sind:

- `numpy` – Eine Bibliothek für schnelles und effizientes Rechnen mit Vektoren und Matrizen.
- `pandas` – Eine Bibliothek zum einfachen Laden von Datendateien wie CSV oder JSON.
- `PySimpleGui` – Eine Bibliothek um interaktive Programme mit einer GUI zu erstellen.

## Module und Packages

Python erlaubt die Gliederung des Source-Codes in Module (einzelne Dateien) und Packages (ein Ordner mit mehreren Dateien, die zu einer Einheit zusammengefasst werden).

...




### Digits-Module

- loading csv files from data
- filtern nach Ziffern oder Arten

### Net-Package

- Klasse mit Layers
- Layers sind separat
- Aktivierungsfunktion
- Error-Funktion

### Hauptprogramm

- main script
- laden von Ziffern
- Netz initialisieren
- trainieren
- Speichern der Simulationen
- Speichern des trainierten Netzes
- => Simulation

### Demo-App

- interaktives GUI Programm
- lädt Netz aus letzter Simulation
- Pixelgrafik => Pixel anklicken to toggle
