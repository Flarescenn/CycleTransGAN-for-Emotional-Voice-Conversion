import torch
import torch.nn as nn
import numpy as np
from scipy.interpolate import interp1d
import pycwt as wavelet
from scipy.signal import firwin, lfilter
import matplotlib.pyplot as plt
from sklearn import preprocessing



def l1_loss(y, y_hat):
    return torch.mean(torch.abs(y - y_hat))

def l2_loss(y, y_hat):
    return torch.mean(torch.square(y - y_hat))

def cross_entropy_loss(logits, labels):
    loss_fn = nn.BCEWithLogitsLoss()  # Binary Cross Entropy + Sigmoid Activation
    return loss_fn(logits, labels)


def convert_continuous_f0(f0):
    """
    Converts a discrete F0 sequence to a continuous F0 sequence.
    Takes a discrete F0 sequence and fills in the "zero" F0 values.

    Args:
        f0 (ndarray): original f0 sequence with the shape (T)

    Return:
        A tuple containing:
      - `uv`: A binary mask indicating voiced (1) and unvoiced (0) frames.
      - `cont_f0`: continuous F0 sequence with the shape (T)
    """
    # Get uv information as binary
    uv = np.float32(f0 != 0)

    if (f0 == 0).all():
        print("Warning: all of the f0 values are 0.")
        return uv, f0

    # Get start and end of f0
    start_f0 = f0[f0 != 0][0]
    end_f0 = f0[f0 != 0][-1]

    # Padding start and end of f0 sequence
    start_idx = np.where(f0 == start_f0)[0][0]
    end_idx = np.where(f0 == end_f0)[0][-1]
    f0[:start_idx] = start_f0
    f0[end_idx:] = end_f0

    # Get non-zero frame index
    nz_frames = np.where(f0 != 0)[0]

    # Perform linear interpolation
    f = interp1d(nz_frames, f0[nz_frames])
    cont_f0 = f(np.arange(0, f0.shape[0]))

    return uv, cont_f0


# Get continuous log-F0 (Normalization purposes)
def get_cont_lf0(f0, frame_period=5.0):
    uv, cont_f0_lpf = convert_continuous_f0(f0)
    cont_lf0_lpf = np.log(cont_f0_lpf)
    return uv, cont_lf0_lpf


# Compute Continuous Wavelet Transform of log-F0
def get_lf0_cwt(lf0):
    mother = wavelet.MexicanHat()
    dt = 0.005
    dj = 1
    s0 = dt * 2
    J = 9
    Wavelet_lf0, scales, freqs, coi, fft, fftfreqs = wavelet.cwt(np.squeeze(lf0), dt, dj, s0, J, mother)
    Wavelet_lf0 = np.real(Wavelet_lf0).T
    return Wavelet_lf0, scales


# Inverse Continuous Wavelet Transform (CWT -> Log F0)
def inverse_cwt(Wavelet_lf0, scales):
    lf0_rec = np.zeros([Wavelet_lf0.shape[0], len(scales)])
    for i in range(0, len(scales)):
        lf0_rec[:, i] = Wavelet_lf0[:, i] * ((i + 1 + 2.5) ** (-2.5))
    lf0_rec_sum = np.sum(lf0_rec, axis=1)
    lf0_rec_sum = preprocessing.scale(lf0_rec_sum)
    return lf0_rec_sum


# Apply low-pass filter
def low_pass_filter(x, fs, cutoff=70, padding=True):
    """FUNCTION TO APPLY LOW PASS FILTER
        (Removing high-frequency noise.)
    Args:
        x (ndarray): Waveform sequence
        fs (int): Sampling frequency
        cutoff (float): Cutoff frequency of low pass filter

    Return:
        (ndarray): Low pass filtered waveform sequence
    """
    nyquist = fs // 2
    norm_cutoff = cutoff / nyquist

    # Low pass filter
    numtaps = 255
    fil = firwin(numtaps, norm_cutoff)
    x_pad = np.pad(x, (numtaps, numtaps), 'edge')
    lpf_x = lfilter(fil, 1, x_pad)
    lpf_x = lpf_x[numtaps + numtaps // 2: -numtaps // 2]
    return lpf_x


# Normalize the Wavelet-transformed F0
def norm_scale(Wavelet_lf0):
    Wavelet_lf0_norm = np.zeros((Wavelet_lf0.shape[0], Wavelet_lf0.shape[1]))
    mean = np.zeros((1, Wavelet_lf0.shape[1]))  # [1,10]
    std = np.zeros((1, Wavelet_lf0.shape[1]))
    for scale in range(Wavelet_lf0.shape[1]):
        mean[:, scale] = Wavelet_lf0[:, scale].mean()
        std[:, scale] = Wavelet_lf0[:, scale].std()
        Wavelet_lf0_norm[:, scale] = (Wavelet_lf0[:, scale] - mean[:, scale]) / std[:, scale]
    return Wavelet_lf0_norm, mean, std


# Denormalize the Wavelet-transformed F0
def denormalize(Wavelet_lf0_norm, mean, std):
    Wavelet_lf0_denorm = np.zeros((Wavelet_lf0_norm.shape[0], Wavelet_lf0_norm.shape[1]))
    for scale in range(Wavelet_lf0_norm.shape[1]):
        Wavelet_lf0_denorm[:, scale] = Wavelet_lf0_norm[:, scale] * std[:, scale] + mean[:, scale]
    return Wavelet_lf0_denorm


# Normalize and scale a batch of F0 sequences
def get_lf0_cwt_norm(f0s, mean, std):
    """
  Processes a list of F0 sequences and computes their normalized CWT coefficients.

  Args:
    f0s: A list of F0 sequences.
    mean: A pre-computed mean for normalization.
    std: A pre-computed standard deviation for normalization.

  Returns:
    A tuple containing:
      - Wavelet_lf0s_norm: A list of normalized CWT coefficients for each F0 sequence.
      - scaless: A list of scales used for the CWT.
      - means: A list of means used for normalization.
      - stds: A list of standard deviations used for normalization.
    """
    uvs = []
    cont_lf0_lpfs = []
    cont_lf0_lpf_norms = []
    Wavelet_lf0s = []
    Wavelet_lf0s_norm = []
    scaless = []
    means = []
    stds = []

    for f0 in f0s:
        uv, cont_lf0_lpf = get_cont_lf0(f0)
        cont_lf0_lpf_norm = (cont_lf0_lpf - mean) / std

        Wavelet_lf0, scales = get_lf0_cwt(cont_lf0_lpf_norm)  # [560, 10]
        Wavelet_lf0_norm, mean_scale, std_scale = norm_scale(Wavelet_lf0)  # [560,10], [1,10], [1,10]

        Wavelet_lf0s_norm.append(Wavelet_lf0_norm)
        uvs.append(uv)
        cont_lf0_lpfs.append(cont_lf0_lpf)
        cont_lf0_lpf_norms.append(cont_lf0_lpf_norm)
        Wavelet_lf0s.append(Wavelet_lf0)
        scaless.append(scales)
        means.append(mean_scale)
        stds.append(std_scale)

    return Wavelet_lf0s_norm, scaless, means, stds
