---
lang: de
documentclass: scrreprt
classoption:
- 10pt                # Schriftgröße 10pt
- a4paper             # Papierformat A4
- bibliography=totoc  # das Literaturverzeichnis in den TOC
- chapterprefix=false # kein Einfügen von "Anhang" bzw. "Kapitel" vor Überschrift
- headings=normal     # kleinere Überschriften verwenden
- headsepline         # Trennlinie zum Seitenkopf Bereich headings
- listof=totoc        # alle Listen in das Inhaltsverzeichnis
- numbers=noenddot    # damit hinter der letzten Ziffer kein Punkt steht (Kapitelnummerierung)
- oneside             # Layout mit einer Spalte
- openright           # einseitiges Layout
- parskip=half-       # Abstand nach Absatz
- plainheadsepline    # Trennlinie zum Seitenkopf Bereich Plain

# toc: true # Inhaltsverzeichnis
# lof: true # Abbildungsverzeichnis
# lot: true # Tabellenverzeichnis

geometry: # Seitenränder
- top=25mm
- bottom=25mm
- left=40mm
- right=30mm
---

\cleardoublepage

\newcounter{exterior}
\setcounter{exterior}{\value{page}}

\pagenumbering{arabic}
