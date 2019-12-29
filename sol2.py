from scipy.io.wavfile import read, write
from scipy import signal
from scipy.ndimage.interpolation import map_coordinates
import scipy.signal
import numpy as np
from skimage.color import rgb2gray
from imageio import imread

CHANGE_SAMPLES_WAV = 'change_samples.wav'

CHANGE_RATE_WAV = 'change_rate.wav'

ROWS = 0

REPRESENTATION_ERROR = "Representation code not exist. please use 1 or 2"

RGB = 2

FILE_PROBLEM = "File Problem"

GREYSCALE = 1

MAX_INTENSITY = 255

der_vec = np.array([0.5, 0, -0.5]).reshape(1, 3)


def representation_check(representation):
    """
    check if representation code is valid
    :param representation: representation code
    """
    if representation not in [GREYSCALE, RGB]:
        print(REPRESENTATION_ERROR)
        exit()


def normalize_0_to_1(im):
    """
    normalize picture
    :param im: image in range 0-255
    :return: image in range [0,1]
    """
    if im.dtype != np.float64:
        im = im.astype(np.float64)
        im /= MAX_INTENSITY
    return im


def read_image(filename, representation):
    """
    This function returns an image, make sure the output image is represented by a matrix of type
    np.float64 with intensities (either grayscale or RGB channel intensities) normalized to the range [0, 1].
    :param filename: the filename of an image on disk (could be grayscale or RGB).
    :param representation: representation code, either 1 or 2 defining whether the output should be a grayscale
    image (1) or an RGB image (2).
    :return: an image
    """
    im = None
    try:
        im = imread(filename)
    except Exception:  # internet didnt have specific documentation regarding the exceptions this func throws
        print(FILE_PROBLEM)
        exit()
    representation_check(representation)
    im = normalize_0_to_1(im)
    if representation == GREYSCALE:
        return rgb2gray(im)
    return im.astype(np.float64)


def stft(y, win_length=640, hop_length=160):
    fft_window = signal.windows.hann(win_length, False)

    # Window the time series.
    n_frames = 1 + (len(y) - win_length) // hop_length
    frames = [y[s:s + win_length] for s in np.arange(n_frames) * hop_length]

    stft_matrix = np.fft.fft(fft_window * frames, axis=1)
    return stft_matrix.T


def istft(stft_matrix, win_length=640, hop_length=160):
    n_frames = stft_matrix.shape[1]
    y_rec = np.zeros(win_length + hop_length * (n_frames - 1), dtype=np.float)
    ifft_window_sum = np.zeros_like(y_rec)

    ifft_window = signal.windows.hann(win_length, False)[:, np.newaxis]
    win_sq = ifft_window.squeeze() ** 2

    # invert the block and apply the window function
    ytmp = ifft_window * np.fft.ifft(stft_matrix, axis=0).real

    for frame in range(n_frames):
        frame_start = frame * hop_length
        frame_end = frame_start + win_length
        y_rec[frame_start: frame_end] += ytmp[:, frame]
        ifft_window_sum[frame_start: frame_end] += win_sq

    # Normalize by sum of squared window
    y_rec[ifft_window_sum > 0] /= ifft_window_sum[ifft_window_sum > 0]
    return y_rec


def phase_vocoder(spec, ratio):
    num_timesteps = int(spec.shape[1] / ratio)
    time_steps = np.arange(num_timesteps) * ratio
    time_steps = time_steps[time_steps < spec.shape[1]]

    # interpolate magnitude
    yy = np.meshgrid(np.arange(time_steps.size), np.arange(spec.shape[0]))[1]
    xx = np.zeros_like(yy)
    coordiantes = [yy, time_steps + xx]
    warped_spec = map_coordinates(np.abs(spec), coordiantes, mode='reflect', order=1).astype(np.complex)

    # phase vocoder
    # Phase accumulator; initialize to the first sample
    spec_angle = np.pad(np.angle(spec), [(0, 0), (0, 1)], mode='constant')
    phase_acc = spec_angle[:, 0]

    for (t, step) in enumerate(np.floor(time_steps).astype(np.int)):
        # Store to output array
        warped_spec[:, t] *= np.exp(1j * phase_acc)

        # Compute phase advance
        dphase = (spec_angle[:, step + 1] - spec_angle[:, step])

        # Wrap to -pi:pi range
        dphase = np.mod(dphase - np.pi, 2 * np.pi) - np.pi

        # Accumulate phase
        phase_acc += dphase

    return warped_spec


def DFT(signal):
    """
    Discrete Fourier Transform
    :param signal: an array of dtype float64 with shape (N,1)
    :return: an array of dtype complex128 with shape (N,1)
    """
    N = signal.shape[ROWS]
    Y = np.arange(N, dtype=np.float64).reshape((N, 1))
    u = Y.T
    powered_e = np.exp(2j * np.pi * (-1) * (1 / N) * (Y @ u))
    return (powered_e @ signal).astype(np.complex128)


def IDFT(fourier_signal):
    """
    Inverse Discrete Fourier Transform
    :param fourier_signal: an array of dtype complex128 with shape (N,1)
    :return: an array of dtype complex128 with shape (N,1)
    """
    N = fourier_signal.shape[ROWS]
    U = np.arange(N, dtype=np.complex128).reshape(N, 1)
    y = U.T
    powered_e = np.exp(2j * np.pi * (1 / N) * (U @ y))
    return (powered_e @ fourier_signal) / N


def DFT2(image):
    """
    convert a 2D discrete signal to its Fourier representation
    :param image: grayscale image of dtype float64
    :return: 2D array of dtype complex128
    """
    return DFT(DFT(image.T).T)


def IDFT2(fourier_image):
    """
    Inverse - convert a 2D discrete signal to its Fourier representation
    :param fourier_image: 2D array of dtype complex128
    :return: 2D array of dtype complex128
    """
    return IDFT(IDFT(fourier_image.T).T)


def change_rate(filename, ratio):
    """
     changes the duration of an audio file by keeping the same samples, but changing the
    sample rate written in the file header
    :param filename:  a string representing the path to a WAV file
    :param ratio: a positive float64 representing the duration change
    """
    curr_rate, sound = read(filename)
    write(CHANGE_RATE_WAV, int(ratio * curr_rate), sound)


def change_samples(filename, ratio):
    """
     changes the duration of an audio file by reducing the number of samples using Fourier
    :param filename:  a string representing the path to a WAV file
    :param ratio: a positive float64 representing the duration change
    :return: a 1D ndarray of dtype float64 representing the new sample points
    """
    rate, sound = read(filename)
    changed_sound = resize(sound, ratio)
    write(CHANGE_SAMPLES_WAV, rate, changed_sound)
    return changed_sound.astype(np.float64)


def resize(data, ratio):
    """
     change the number of samples by the given ratio
    :param data: 1D ndarray of dtype float64 or complex128(*) representing the original sample points
    :param ratio: a positive float64 representing the duration change
    :return:
    """
    curr_size = len(data)
    new_size = int(np.floor(curr_size / ratio))
    fourier_signal = DFT(data.reshape(curr_size, 1))
    centered_fourier_signal = np.fft.fftshift(fourier_signal)
    if new_size > curr_size:
        num_of_zeros_per_side = (new_size - curr_size) / 2
        resized_data = np.zeros((new_size,), dtype=np.complex128).reshape(new_size, 1)
        resized_data[
        int(np.ceil(num_of_zeros_per_side)):-int(np.floor(num_of_zeros_per_side))] = centered_fourier_signal
    elif new_size == curr_size:
        resized_data = centered_fourier_signal
    else:
        num_of_sample_clip_per_side = (curr_size - new_size) / 2
        resized_data = centered_fourier_signal[
                       int(np.ceil(num_of_sample_clip_per_side)):-int(np.floor(num_of_sample_clip_per_side))]
    final_result = IDFT(np.fft.ifftshift(resized_data))
    final_result = final_result.reshape((1, new_size))[0]
    data_type = data.dtype
    if data_type != np.complex128:
        final_result = np.real(final_result)
    return final_result.astype(data_type)


def resize_spectrogram(data, ratio):
    """
     speeds up a WAV file, without changing the pitch, using spectrogram scaling
    :param data: 1D ndarray of dtype float64 or complex128(*) representing the original sample points
    :param ratio: a positive float64 representing the duration change
    :return:
    """
    spectrogram = stft(data)
    resized_spectrogram = np.apply_along_axis(resize, 1, spectrogram, ratio)
    return istft(resized_spectrogram).astype(data.dtype)


def resize_vocoder(data, ratio):
    """
    speedups a WAV file by phase vocoding its spectrogram
    :param data: 1D ndarray of dtype float64 or complex128(*) representing the original sample points
    :param ratio: a positive float64 representing the duration change
    :return:
    """
    return istft(phase_vocoder(stft(data), ratio)).astype(data.dtype)


def magnitude(dx, dy):
    """
    calc magnitude (standard norm)
    :param dx: one direction np shape
    :param dy: other direction np shape
    :return: magnitude np shape
    """
    return np.sqrt(np.abs(dx) ** 2 + np.abs(dy) ** 2).astype(np.float64)


def conv_der(im):
    """
    computes the magnitude of image derivatives
    :param im: grayscale images of type float64
    :return: grayscale images of type float64
    """
    dx = scipy.signal.convolve2d(im, der_vec, mode='same')
    dy = scipy.signal.convolve2d(im, der_vec.T, mode='same')
    return magnitude(dx, dy)


def fourier_der(im):
    """
    computes the magnitude of image derivatives using Fourier transform
    :param im: grayscale images of type float64
    :return: grayscale images of type float64
    """
    M, N = im.shape
    fourier_transform = np.fft.fftshift(DFT2(im))
    const = 2j * np.pi
    u = np.arange(np.ceil(-M / 2), np.ceil(M / 2)).reshape(M, 1) * const / M
    v = np.arange(np.ceil(-N / 2), np.ceil(N / 2)).reshape(1, N) * const / N
    dx = IDFT2(np.fft.ifftshift(fourier_transform * u))
    dy = IDFT2(np.fft.ifftshift(fourier_transform * v))
    return magnitude(dx, dy)
