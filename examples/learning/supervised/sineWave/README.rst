===========
 Sine Wave
===========

These are simple examples of neural networks to approximate sine wave like functions:

- run-ffn.py approximate a function without a time-dependence:
    A simple ffnn tries to learn a function for a training set of (x, y) where the y's are the labels.

    .. math::
        y(x)=\tanh(\exp(\sin(\texttt{trainingInputSet})))\cdot\texttt{scaling}

- run-rnn.py approximate a function with a time-dependence:
   | A simple rnn tries to learn a function for a training set of (x(t), y(t)) values where the y(T)'s are the labels.
   | The target is a single ouput value :math:`y(x, T)`.
   | The training sequences may have different length's i.e. T is not fixed but they are limited max :math:`T<T_{\text{final}}`.

    .. math::
        y(x,t)=\sin(x\cdot\texttt{peak} + w \cdot t)

    - :math:`\texttt{trainingInputSetX\left[x_{idx}\right]\left[t_{idx}\right]\left[\left[x(x_{idx}, t_{idx})\right]\right]}` where
        - :math:`\texttt{trainingInputSetX\left[x_{idx}\right]}` is a list of :math:`T_{x_idx}` lists, of input features :math:`x(x_{idx}, t_{x_idx})`.
    - :math:`\texttt{trainingInputSetT\left[x_{idx}\right]\left[t_{idx}\right]\left[\left[t(x_{idx}, t_{idx})\right]\right]}`
        - :math:`\texttt{trainingInputSetT\left[x_{idx}\right]}` is a list of :math:`T_{x_idx}` lists, of time points :math:`t(x_{idx}, t_{idx})` with :math:`t(x_{idx}, t_{idx})<=T(x_{idx}, t_{idx})<=T_{max}`
    - :math:`\texttt{trainingSolutionSet\left[x_{idx}\right]\left[\left[y(x_{idx}, T)\right]\right]}`
