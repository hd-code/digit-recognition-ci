# Lösungsmethode

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
