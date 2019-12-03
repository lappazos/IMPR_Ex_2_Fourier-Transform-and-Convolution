import numpy as np
from scipy.io.wavfile import read, write
import ex2_helper


def DFT(signal):
    N = signal.shape[0]
    Y = np.arange(N, dtype=np.float64).reshape((N, 1))
    u = Y.T
    powered_e = np.exp(2j * np.pi * (-1) * (1 / N) * (Y @ u))
    return powered_e @ signal


def IDFT(fourier_signal):
    N = fourier_signal.shape[0]
    U = np.arange(N, dtype=np.complex128).reshape(N, 1)
    y = U.T
    powered_e = np.exp(2j * np.pi * (1 / N) * (U @ y))
    return np.real_if_close((powered_e @ fourier_signal) / N)


def DFT2(image):
    return DFT(DFT(image.T).T)


def IDFT2(fourier_image):
    return IDFT(IDFT(fourier_image.T).T)


def change_rate(filename, ratio):
    curr_rate, sound = read(filename)
    write('change_rate.wav', int(ratio * curr_rate), sound)


def change_samples(filename, ratio):
    rate, sound = read(filename)
    write('change_samples.wav', rate, resize(sound, ratio))


def resize(data, ratio):
    curr_size = len(data)
    new_size = np.floor(curr_size / ratio)
    if new_size % 2 != 0:
        new_size = new_size - 1
    fourier_signal = DFT(data.reshape(curr_size, 1))
    centered_fourier_signal = np.fft.fftshift(fourier_signal)
    if new_size > curr_size:
        num_of_zeros_per_side = int((new_size - curr_size) / 2)
        resized_data = np.zeros((new_size,))
        resized_data[num_of_zeros_per_side:-num_of_zeros_per_side] = centered_fourier_signal
    else:
        num_of_sample_clip_per_side = int((curr_size - new_size) / 2)
        resized_data = centered_fourier_signal[num_of_sample_clip_per_side:-num_of_sample_clip_per_side]
    return IDFT(np.fft.ifftshift(resized_data)).reshape((new_size,))

def resize_spectrogram(data, ratio):




# def iDFT(signal):
#     x = np.transpose([np.arange(0,signal.shape[0])])
#     mat = np.exp((-2j*np.pi*(x@x.T))/signal.shape[0])
#     dft = mat@signal
#     # assert np.isclose(dft,np.fft.fft(signal)).all()
#     return dft


h = np.array([2, 3, 3], dtype=np.float64).reshape(3, 1)
#
#
# print(DFT(h))
#
h = np.array([[0.3, 0.3, 0.59], [0.22, 0.98, 0.03]], dtype=np.float64)
#
#
# print(IDFT(DFT(h)))
rate, lior = read('external\\aria_4kHz.wav')
print(IDFT2(DFT2(h)))
# print(np.fft.fft(h))
