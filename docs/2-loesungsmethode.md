# Lösungsmethode

## Überwachtes Lernen

Lösung mit Überwachtem Lernen
Traingsdaten => beispielhafte Input und Output-Werte
Training eines selbstlernenden Algorithmus
=> liefert am Ende zu den Inputs die richtigen Outputs
und ist außerdem in der Lage beliebige Inputs zu verarbeiten und richtige Outputs zu liefern

Traingsdaten => Sets von 0-9: normal, klein, digital, digital-klein
Testdaten => evag, digital-klein-unten

## Neuronale Netze

Orientiert an Gehirn => Neuronen sind verbundene Nervenzellen...

Einzelnes Neuron orientiert sich an Perzeptron

$$
p = a(\vec w * \vec i + b)
$$

Schichten aus Neuronen => vollvermascht

$$
\vec o = a(W * \vec i + \vec b)
$$

$$
\vec i_i = \vec o_{i-1}
$$

Eine versteckte Schicht reicht aus um beliebige Funktionen zu approximieren

$$
\vec o = a(\vec b_o + W_o * a(\vec b_h + W_h * \vec i))
$$


## Gradientenabstiegsverfahren

Trainingsverfahren

Batch-Learning






Die Lösung erfolgt über das sog. Überwachte Lernen (QUELLE). Das heißt, das ein Algorithmus wird zunächst mit einer Reihe von Trainingsdatensätzen trainiert. Diese Trainingsdatensätzen beinhalten einen Input und den dazu geforderten Output. Mittels des Trainings ist der Algorithmus am Ende in der Lage, zu den Trainingsdaten den richtigen Output zurückzugeben. Die Idee ist nun, dass der Algorithmus während des Trainings eine Abstraktion durchführt und somit in der Lage ist, Inputwerte zu verarbeiten, die es noch nicht gesehen hat...





Ziffern erkennen

Beschreibung der Datensätze
...

überwachtes lernen




Perzeptron

$$
p = sig(\vec w * \vec i + b)
$$

$$
sig(x) = \frac{1}{1+e^{-x}} \\
sig'(x) = sig(x) * (1 - sig(x))
$$

mehrere Perzeptrons

Vorwärtsberechnung

$$
\vec h = sig(\vec b_h + W_h * \vec i) \\
\vec o = sig(\vec b_o + W_o * \vec h) \\
\vec o = sig(\vec b_o + W_o * sig(\vec b_h + W_h * \vec i))
$$

Lernverfahren

Gradientenabstiegsverfahren

Fehlerfunktion

Mean Squared Error

$$
mse(\vec o, \vec t) = \frac{1}{n} * \sum (o_i - t_i)^2 \\
mse'_i(\vec o, \vec t) = \frac{2}{n} * (o_i - t_i)
$$

Backpropagation

\begin{align*}
\delta b_o &= mse'(\vec o, \vec t) * sig'(\vec b_o + W_o * \vec h) \\
\delta W_o &= mse'(\vec o, \vec t) * sig'(\vec b_o + W_o * \vec h) *  \vec h^T \\ \\
\delta b_h &= mse'(\vec o, \vec t) * sig'(\vec b_o + W_o * \vec h) * W_o^T * sig'(\vec b_h + W_h * \vec i) \\
\delta W_h &= mse'(\vec o, \vec t) * sig'(\vec b_o + W_o * \vec h) * W_o^T * sig'(\vec b_h + W_h * \vec i) * \vec i^T
\end{align*}






perzeptron netz, Bild der Pixel => Input => Hidden => Output

Vorwärtsberechnung => sigmoid sig(b + W * i)

Gradientenabstiegsverfahren

- mean squared error
- backpropagation ...
