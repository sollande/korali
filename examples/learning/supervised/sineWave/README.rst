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

    - :math:`\texttt{trainingInputSetX\left[x_idx\right]\left[t_idx\right]\left[\left[x(x_idx, t_idx)\right]\right]}`
    - :math:`\texttt{trainingInputSetT\left[x_idx\right]\left[t_idx\right]\left[\left[t(x_idx, t_idx)\right]\right]}`
    - :math:`\texttt{trainingSolutionSet\left[x_idx\right]\left[\left[y(x_idx, T)\right]\right]}`
