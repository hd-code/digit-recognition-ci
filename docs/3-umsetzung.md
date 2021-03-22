# Umsetzung in Python

Dieses Projekt befindet sich in einem Git-Repository. Dort befinden sich alle Dateien des Projektes, inklusive Source-Code, Dokumentation usw. Das Repository ist unter diesem Link erreichbar: <https://github.com/hd-code/digit-recognition-ci/>

## Installation

Das Projekt ist in Python umgesetzt. Zusätzlich ist der Package-Manager Pipenv verwendet worden, um alle benötigten Software-Komponenten zu installieren und zu verwalten.

### Vorraussetzungen

Die Vorraussetzungen um das Projekt ausführen zu können sind:

- Das Projekt inklusive aller Dateien muss sich auf dem Computer befinden.
- **Python** ab *Version 3.9 oder höher* – Informationen zur Installation gibt es hier:  
  <https://www.python.org>
- **Pipenv** ab *Version 2020.11 oder höher* – Installation ist über folgenden Befehl im Terminal möglich: `pip install pipenv`

*Hinweis:* Damit der Befehl zur Installation von Pipenv funktioniert, muss Python bereits auf dem System installiert sein.

### Installation der externen Software-Komponenten

Nun muss ein Terminal im Ordner des Projektes geöffnet werden. Durch die Ausführung des folgenden Befehls werden alle benötigten Software-Komponenten automatisch installiert:

```sh
pipenv install
```

Das ist alles. Nun kann das Projekt verwendet werden.

## Externe Software-Komponenten

Wie die benötigten Software-Komponenten installiert werden, ist im vorherigen Abschnitt erklärt worden. Hier folgt eine Auflistung, was an externen Software-Komponenten benutzt worden ist und warum.

- `numpy` – eine Bibliothek für schnelles und effizientes Rechnen mit Vektoren und Matrizen.
- `pandas` – eine Bibliothek zum einfachen Speichern, Laden und Visualisieren von tabellarischen Daten.
- `matplotlib` – wird von `pandas` benötigt, um Daten in Form von Grafiken visualisieren zu können.
- `jupyter` – ist eine Entwicklungsumgebung, in der Codeschnipsel geschrieben, ausgeführt und die Ergebnisse direkt angezeigt werden können.
- `ipykernel` – wird von `jupyter` benötigt, um Python-Code ausführen zu können.
- `PySimpleGui` – eine Bibliothek, um interaktive Programme mit einer GUI zu erstellen.

## Projektübersicht

### Projektstruktur

Damit das Projekt übersichtlich bleibt, ist es in Ordner und Unterordner gegliedert. Die Ordner sind:

- `data`
  - `cache` – enthält die Ergebnisse einer Simulation
  - `digits` – enthält die Ziffern für die Trainings- und Testdaten als CSV-Dateien
- `docs` – Dokumentation des Projektes (also faktisch dieses Dokument).
- `src` – Source-Code, weitere Erklärung in den folgenden Abschnitten

### Hauptprogramm

Im Hauptprogramm wird das neuronale Netz initialisiert, trainiert und analysiert. Die entsprechende Datei ist `src > main.ipynb`.

Zum komfortableren Arbeiten ist hierfür ein **Jupyter Notebook** verwendet worden. Ein Jupyter Notebook ist ein Dokument, welches im Webbrowser geöffnet und bearbeitet werden kann. Man hat hier sog. Code-Zellen. In diese kann Code eingefügt und direkt ausgeführt werden. Die Ergebnisse werden direkt unter der Zelle dargestellt. Ändert man den Code in der Zelle und führt ihn erneut aus, so werden die Ergebnisse entsprechend aktualisiert. Dadurch eignen sich Jupyter Notebooks optimal für das maschinelle Lernen. Es ist sehr einfach Simulationen mit verschiedenen Parametern auszuführen bis die optimalen Werte gefunden sind.

Vor der Verwendung von Jupyter muss der folgende Befehl im Terminal im Ordner des Projektes ausgeführt:

```sh
pipenv run jupyter notebook
```

Nun muss im Webbrowser die folgende Seite aufgerufen werden: <http://localhost:8888>. Jetzt kann man zur besagten Datei navigieren, diese öffnen und bearbeiten.

Zuletzt noch einige Hinweise zu der Datei: Wenn in den Code-Zellen eine Variable ausschließlich in Großbuchstaben geschrieben ist (z.B. `DIGITS`), dann wird diese Variable auch in anderen Code-Zellen der Datei verwendet. Die Zelle, wo die Variable definiert wird, muss also auf jeden Fall ausgeführt werden, bevor eine der folgenden Zellen ausgeführt werden kann. Die generierten Daten werden im Ordner `data > cache` abgespeichert (in den entsprechenden Code-Zellen befinden sich Kommentare). Dadurch können die Ergebnisse der letzten Simulation geladen werden und das zeitintensive Training muss nicht jedes mal erneut durchgeführt werden.

### Demo-App

Mit der Demo-App kann eine interaktive Anwendung gestartet werden, um das generierte neuronale Netz zu testen. Die Implementierung befindet sich in der Datei `src > app.py`.

Zum Starten der App bitte folgenden Befehl im Terminal im Ordner des Projektes ausführen:

```sh
pipenv run python src/app.py
```

Es erscheint ein Fenster mit einem Feld aus Pixeln auf der linken und der Ausgabe des neuronalen Netzes auf der rechten Seite. Die Pixel können angeklickt werden. Dadurch wird ein Pixel von leer zu gefüllt geändert und umgekehrt. Nach einem Klick wird sofort die Berechnung über das Netz durchgeführt und das Ergebnis angezeigt.

### Hilfsmodule und -packages

Python erlaubt die Gliederung des Source-Codes in Module (einzelne Dateien) und Packages (ein Ordner mit mehreren Dateien, die zu einer Einheit zusammengefasst werden). Dies wird benutzt, um verschiedene Teile des Programms in wiederverwendbare Komponenten auszulagern.

#### Digits-Module

Dieses Module lädt die Ziffern (also die Beispieldaten), welche sich im Ordner `data > digits` als CSV-Dateien befinden. Die Implementierung befindet sich in der Datei `src > digits.py`. Es ist möglich die Ziffern zu filtern (nach der Ziffer selbst oder nach dem Set, zu welchem eine Ziffer gehört).

Die Ziffern sind so gespeichert, dass der Dateiname mit der Ziffer beginnt, welche sich hinter der Datei verbirgt. Es folgt der Name des Ziffern-Sets mit Bindestrichen getrennt (näheres im nächsten Kapitel). So lautet der Dateiname für die Ziffer 5 des evag Datensets z.B. `5-evag.csv`.

#### Net-Package

Dieses Package implementiert das neuronale Netz als wiederverwendbare Bibliothek. Es besteht aus mehreren Dateien, die sich alle im Ordner `src > net` befinden. Alle Teilaspekte eines neuronalen Netzes finden sich hier wieder (Aktivierungs- und Fehlerfunktionen, die Schichten sowie die Möglichkeit ein Netz zu speichern und zu laden).

Die Datei `__init__.py` legt fest, welche Methoden dieses Package nach außen bereitstellt. Es gibt Methoden, um ein Netz mit verschiedenen Neuronen zu initialisieren (`init`), Berechnungen durchzuführen (`calc` und `calcBatch`), Fehlerwerte zu ermitteln (`calcError` und `calcBatchError`) und natürlich ein Netz zu trainieren (`train` und `trainBatch`). Zusätzlich kann ein Netz auf der Festplatte gespeichert (`save`) und wieder geladen werden (`load`).

#### Testing

Die Hilfsmodule müssen ordentlich funktionieren, damit sie bedenkenlos eingebunden und wiederverwendet werden können. Deshalb befindet sich am Ende von allen diesen Dateien ein Abschnitt, welcher den Code entsprechend testet. Um eine Datei zu testen, muss sie direkt im Terminal ausgeführt werden. Dies geht im Projektordner im Terminal über folgenden Befehl:

```sh
PYTHONPATH=src pipenv run python src/<path-to-file>.py
```

Sollte es bei der Ausführung zu Fehlern kommen, so werden diese im Terminal angezeigt. Andernfalls ist am Ende `SUCCESS` zu sehen.

Mit folgendem Befehl können auch alle Hilfsmodule auf einmal getestet werden:

```sh
files=(src/digits.py src/net/[^_]*.py)
for f in $files; do PYTHONPATH=src pipenv run python $f; done
```
