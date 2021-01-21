# Ziffern erkennen – CI

Dieses Repo enthält die Projektabgabe für das Fach "Computational Intelligence" an der Fachhochschule Erfurt. Ziel des Projektes ist es, ein neuronales Netz zu programmieren, welches in der Lage ist, die Ziffern von 0 bis 9 in kleinen Pixelrastern zu erkennen.

## Installation

Die Skripte des Projektes sind in Python geschrieben. Es muss also Python ab Version 3 installiert sein.

## Projektstruktur

## Testing

In allen Dateien in `src` sind am Ende ein paar kleine Tests geschrieben, mit denen überprüft werden kann, ob die Funktionen richtig arbeiten.

Um die Tests auszuführen, müssen die Dateien direkt mit Python als Main-Skripte ausgeführt werden. Z.B. in Bash-Shells kann das mit folgendem Befehl bewerkstelligt werden:

```sh
for f in src/*.py; do python $f; done
```