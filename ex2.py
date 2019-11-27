import numpy as np


def DFT(signal):
    N = signal.shape[0]
    X = np.arange(N, dtype=np.float64).reshape(signal.shape)
    u = X.T
    powered_e = np.exp(2j * np.pi * (-1) * (1 / N)*(X @ u))
    return powered_e @ signal

    # power = (-2j*np.pi*)
    # return np.exp()


def IDFT(fourier_signal):
    pass


print(DFT(np.array([2, 3, 3]).reshape(3, 1)))
print(np.fft.fft(np.array([2, 3, 3]).reshape(3, 1)))
