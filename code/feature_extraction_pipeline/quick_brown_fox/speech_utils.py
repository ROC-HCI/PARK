# Copyright © CoML 2020, Licensed under the EUPL

import scipy.signal as ss
from typing import Optional, Tuple


from matplotlib import mlab

import matplotlib.pyplot as plt

import numba as nb

import numpy as np

import scipy

import scipy.signal

import sympy

from scipy.stats import entropy

print("Imported all necessary libraries for feature extraction")

# Global variables

primetable = np.concatenate(
    [np.array([1]), np.array(list(sympy.primerange(0, 10000)))])


@nb.jit(nb.int32[:](nb.float32[:, :], nb.float32, nb.int32),

        nopython=True)
def recurrence_histogram(ts: np.ndarray, epsilon: float, t_max: int):

    # print(ts.shape)
    """Numba implementation of the recurrence histogram described in

    http://www.biomedical-engineering-online.com/content/6/1/23

    Parameters

    ----------

    ts: np.ndarray

        Time series of dimension (T,N)

    epsilon: float:

        Recurrence ball radius

    t_max: int

        Maximum distance for return.

        Larger distances are not recorded. Set to -1 for infinite distance.

    Returns

    -------

    recurrence_histogram: np.ndarray

        Histogram of return distances

    """

    return_distances = np.zeros(len(ts), dtype=np.int32)

    for i in np.arange(len(ts)):

        # finding the first "out of ball" index

        first_out = len(ts)  # security

        for j in np.arange(i + 1, len(ts)):

            if 0 < t_max < j - i:

                break

            d = np.linalg.norm(ts[i] - ts[j])

            if d > epsilon:

                first_out = j

                break

        # finding the first "back to the ball" index

        for j in np.arange(first_out + 1, len(ts)):

            if 0 < t_max < j - i:

                break

            d = np.linalg.norm(ts[i] - ts[j])

            if d < epsilon:

                return_distances[j - i] += 1

                break

    return return_distances


# def embed_time_series(data: np.ndarray,dim: int,tau: int):

#     embed_elements = data.shape[0] - (dim - 1) * tau

#

#     y = np.empty((dim*embed_elements,1))

#     for d in range(0,dim):

#         inputDelay = (dim - d - 1) * tau

#         for i in range(0,embed_elements):

#             y[i * dim + d] = data[i + inputDelay]

#     y = y.reshape(embed_elements,-1)

#     y = np.float32(y)

#     return y


def embed_time_series(data: np.ndarray, dim: int, tau: int):
    """Embeds the time series, tested against the pyunicorn implementation

    of that transform"""

    embed_points_nb = data.shape[0] - (dim - 1) * tau

    # embed_mask is of shape (embed_pts, dim)

    embed_mask = np.arange(embed_points_nb)[:, np.newaxis] + np.arange(dim)

    tau_offsets = np.arange(dim) * (tau - 1)  # shape (dim,1)

    embed_mask = embed_mask + tau_offsets

    return data[embed_mask]


def rpde(time_series: np.ndarray,

         dim: int = 4,

         tau: int = 35,

         epsilon: float = 0.12,

         tmax: Optional[int] = None) -> Tuple[float, np.ndarray]:
    """

    Parameters

    ----------

    time_series: np.ndarray

        The input time series. Has to be float32, normalized to [-1,1]

    dim: int

        The dimension of the time series embeddings.

        Defaults to 4

    tau: int

        The "stride" between each of the embedding points in a time series'

        embedding vector. Should be adjusted depending on the

        sampling rate of your input data.

        Defaults to 35.

    epsilon: float

        The size of the unit ball described in the RPDE algorithm.

        Defaults to 0.12.

    tmax: int, optional

        Maximum return distance (n1-n0), return distances higher than this

        are ignored. If set, can greatly improve the speed of the distance

        histogram computation (especially if your input time series has a lot of points).

        Defaults to None.

    parallel: boolean, optional

        Use the parallelized Numba implementation. The parallelization overhead

        might make this slower in cases where the time series is very short.

        Defaults to True.

    Returns

    -------

    rpde: float

        Value of the RPDE

    histogram: np.ndarray

        1-dimensional array corresponding to the histogram of the return

        distances

    """

    # (RPDE expects the array to be floats in [-1,1]

    if not (time_series.dtype == np.float32):

        raise ValueError("Time series should be float32")

    if not (np.abs(time_series) <= 1.0).all():

        raise ValueError("The time series' values have to be normalized "

                         "to [-1, 1]")

    embedded_ts = embed_time_series(time_series, dim, tau)

    # embedded_ts = embed_time_series2(time_series, dim, tau)

    # "flattening" all dimensions but the first (the number of embedding vectors)

    # this doesn't change the distances computed in the recurrence histogram,

    # as it just changes the way the embedding vector are presented, not their values

    if len(embedded_ts.shape) > 2:

        embedded_ts = embedded_ts.reshape(embedded_ts.shape[0], -1)

    # print(embedded_ts.shape)

    histogram_fn = recurrence_histogram

    if tmax is None:

        rec_histogram = histogram_fn(embedded_ts, epsilon, -1)

        # tmax is the highest non-zero value in the histogram

        if rec_histogram.max() == 0.0:

            tmax_idx = 0

        else:

            tmax_idx = np.argwhere(rec_histogram != 0).flatten()[-1]

    else:

        rec_histogram = histogram_fn(embedded_ts, epsilon, tmax)

        tmax_idx = tmax

    # culling the histogram at the tmax index

    culled_histogram = rec_histogram[:tmax_idx]

    # print(culled_histogram.shape)

    # normalizing the histogram (to make it a probability)

    # and computing its entropy

    if culled_histogram.sum():

        normalized_histogram = culled_histogram / culled_histogram.sum()

        histogram_entropy = entropy(normalized_histogram)

        return histogram_entropy / np.log(culled_histogram.size), culled_histogram

    else:

        return 0.0, culled_histogram


def pitchStrengthOneCandidate(f, NL, pc):

    n = (int)(np.fix(f[len(f)-1]/pc - 0.75))

    if n == 0:

        return None

    # f = f.reshape(-1,1)

    k = np.zeros(f.shape)

    q = (f/pc)

    indices = primetable[primetable <= n]

    for i in indices:

        a = np.abs(q-i)

        p = np.argwhere(a < 0.25)

        k[p] = np.cos(2*np.pi*q[p])

        v = np.argwhere(0.25 < a)

        v2 = np.argwhere(a < 0.75)

        v = np.intersect1d(v, v2)

        k[v] = k[v] + np.cos(2*np.pi*q[v])

    k = np.multiply(k, np.sqrt(1/f))

    tmp = np.linalg.norm(k[k > 0], 2)

    k = k/tmp

    S = np.dot(k.transpose(), NL)

    return S


def pitchStrengthAllCandidates(f, L, pc):

    S = np.zeros((len(pc), L.shape[1]))

    k = np.zeros((1, len(pc)+1), dtype=int).flatten()

    for j in range(0, len(k)-1):

        k[j+1] = k[j]+np.where(f[(int)(k[j]):len(f)] > (pc[j]/4))[0][0]

    k = k[1:len(k)].reshape((1, -1))

    L_mult = np.multiply(L, L)

    N = np.sqrt(np.flipud(np.cumsum(np.flipud(L_mult)))).reshape(L.shape)

    for j in range(0, len(pc)):

        n = N[k[0, j], :]

        n[n == 0] = np.inf

        n = n.reshape(1, -1)

        rowIdx = np.arange(0, n.shape[0], 1).transpose().reshape(-1, 1)

        colIdx = np.arange(0, n.shape[1], 1).transpose().reshape(-1, 1)

        tmp = n[rowIdx[:, np.zeros(
            (L.shape[0]-k[0][j], 1), dtype=int).flatten()].flatten()]

        divisor = tmp[:, colIdx.flatten()]

        dividend = L[k[0, j]:L.shape[0]]

        NL = np.divide(dividend, divisor)

        S[j, :] = pitchStrengthOneCandidate(f[k[0, j]:, 0], NL, pc[j])

    return S


def hz2erbs(hz):

    erbs = 6.44*(np.log2(229+hz) - 7.84)

    return erbs


def erbs2hz(erbs):

    hz = np.power(2, (erbs/6.44 + 7.84)) - 229

    return hz


def swipep(x, fs, plo=30, phi=5000, dt=0.001, dlog2p=(1.0/48.0), dERBs=0.1, woverlap=0.5, sTHR=float('-inf')):

    t = np.arange(0, x.shape[0]/fs, dt).reshape((-1, 1))

    # Define pitch candidates

    log2pc = np.arange(np.log2(plo), np.log2(phi), dlog2p).reshape(-1, 1)

    pc = np.power(2, log2pc)

    S = np.zeros((pc.shape[0], t.shape[0]))

    # Determine P2-WSs

    Ws = np.array([(8*fs)/plo, (8*fs)/phi])

    logWs = np.round(np.log2(Ws))

    ws = np.power(2, np.arange(logWs[0], logWs[1]-1, -1))

    pO = (8*fs)/ws

    # Determine window sizes used by each pitch candidate

    d = log2pc - np.log2((8*fs)/ws[0])

    # Create ERB-scale uniformly spaced frequencies (Hz)

    fERBs = erbs2hz(np.arange(hz2erbs(np.min(pc)/4),
                    hz2erbs(fs/2), dERBs)).reshape(-1, 1)

    for i in range(0, len(ws)):

        dn = max(1, np.round(8*(1-woverlap)*fs/pO[i]))

        # Zero pad signal

        xzp0 = np.zeros((int(ws[i]/2.0), 1)).flatten()

        xzp1 = x

        xzp2 = np.zeros((int(dn+ws[i]/2), 1)).flatten()

        xzp = np.concatenate((xzp0, xzp1, xzp2)).reshape(-1, 1)

        # Hann implementation

        window_size = ws[i]

        half_window_size = np.round(window_size/2)

        T = np.arange(1, half_window_size+1)

        w = 0.5 * (1-np.cos((2*np.pi*T)/(window_size+1)))

        w_rev = np.flip(w)

        w = np.concatenate((w, w_rev))

        # Compute specgram

        # w = scipy.signal.windows.hann(int(ws[i]));

        # print(max(w))

        # print(w.shape)

        o = max(0, round(ws[i]-dn))

        # Matched exactly with Max until here

        # f, ti, X = scipy.signal.spectrogram(xzp.flatten(),fs,w,ws[i],o,mode='complex')

        X, f, ti = mlab.specgram(x=xzp.flatten(), NFFT=(int)(
            ws[i]), Fs=fs, window=w, noverlap=(int)(o), mode='complex')

        f = f.reshape(-1, 1)

        ti = ti.reshape(-1, 1)

        if len(ws) == 1:

            # print("Case 1 for iteration %d"%(i))

            j = np.transpose(pc)[:, 0]

            k = np.array([])

            # print(k.shape)

        elif i == len(ws)-1:

            # print("Case 2 for iteration %d" % (i))

            j = np.argwhere(d-i > -1)[:, 0]

            k = np.argwhere(d[j]-i < 0)[:, 0]

            # print(k.shape)

        elif i == 0:

            # print("Case 3 for iteration %d" % (i))

            j = np.argwhere(d-i < 1)[:, 0]

            k = np.argwhere(d[j]-i > 0)[:, 0]

            # print(k.shape)

        else:

            # print("Case 4 for iteration %d" % (i))

            j = np.argwhere(np.abs(d-i) < 1)[:, 0]

            k = np.arange(0, len(j))

            # print(k.shape)

        temp = np.where(fERBs > (pc[j[0]]/4))[0][0]

        fERBs = fERBs[temp:len(fERBs)]

        intp = scipy.interpolate.interp1d(
            f.flatten(), np.abs(X[:, 0]).flatten(), kind='zero')

        temp = intp(fERBs)

        interp_data = np.empty((temp.shape[0], X.shape[1]))

        for idx in range(0, X.shape[1]):

            # intp = scipy.interpolate.interp1d(f.flatten(), np.abs(X[:, idx]).flatten(), kind='zero')

            intp = scipy.interpolate.interp1d(
                f.flatten(), np.abs(X[:, idx]).flatten())

            interp_data[:, idx] = intp(fERBs).flatten()

        interp_data = np.sqrt(np.maximum(0, interp_data))

        # Compute pitch strength

        Si = pitchStrengthAllCandidates(fERBs, interp_data, pc[j])

        if Si.shape[1] > 1:

            Si = Si.transpose()

            intp = scipy.interpolate.interp1d(
                ti.flatten(), Si[:, 0].flatten(), kind='linear', fill_value='extrapolate')

            temp = intp(t)

            interp_data = np.empty((temp.shape[0], Si.shape[1]))

            for idx in range(0, Si.shape[1]):

                intp = scipy.interpolate.interp1d(
                    ti.flatten(), Si[:, idx].flatten(), kind='linear', fill_value='extrapolate')

                # intp = scipy.interpolate.interp1d(ti,Si,kind='linear')

                interp_data[:, idx] = intp(t).flatten()

            Si = interp_data.transpose()

        else:

            Si = np.tile(None, (Si.shape[0], t.shape[0]))

        lmbda = d[j[k]]-i

        mu = np.ones(j.shape).flatten()

        mu[k] = 1 - np.abs(lmbda.flatten())

        temp = np.tile(mu.reshape(-1, 1), (1, Si.shape[1]))

        S[j, :] = S[j, :] + np.multiply(temp, Si)

    # Continue from here

    # Fine tune pitch using parabolic interpolation

    p = np.tile(None, (S.shape[1], 1)).flatten()

    s = np.tile(None, (S.shape[1], 1)).flatten()

    pc = pc.flatten()

    for j in range(0, S.shape[1]):

        s[j] = np.max(S[:, j])

        i = np.where(S[:, j] == s[j])[0][0]

        if s[j] < sTHR:

            continue

        if (i == 0 or i == len(pc)-1):

            p[j] = pc[i]

        else:

            I = np.arange((i-1), (i+2), 1)

            tc = 1/pc[I]

            ntc = ((tc/tc[1])-1)*2*np.pi

            c = np.polyfit(ntc, S[I, j], 2)

            temp = np.arange(np.log2(pc[I[0]]), np.log2(
                pc[I[2]]+1/12/100), 1/12/100)

            ftc = 1/np.power(2, temp)

            nftc = ((ftc/tc[1])-1)*2*np.pi

            pval = np.polyval(c, nftc)

            s[j] = np.max(pval)

            k = np.where(pval == s[j])[0][0]

            p[j] = np.power(2, np.log2(pc[I[0]])+(k/12/100))

    pc = pc.reshape(-1, 1)

    return (p, t, s)


def ppe(pitch_values, f0min=50, f0max=500, unit="Hertz"):

    Nf = 20

    df = (f0max - f0min) / (Nf - 1)

    f = np.arange(f0min, f0max + 1, df)

    n = np.histogram(pitch_values, f)[0]

    p = n / n.sum()

    p[p == 0] = 1

    lp = np.log(p)

    ppe = -np.multiply(p, lp).sum() / np.log(Nf)

    return ppe


def hann(x):

    window_size = x

    half_window_size = np.round(window_size / 2)

    T = np.arange(1, half_window_size + 1)

    w = 0.5 * (1 - np.cos((2 * np.pi * T) / (window_size + 1)))

    w_rev = np.flip(w)

    w = np.concatenate((w, w_rev))

    return w


def hamming(n):

    if n % 2 == 0:

        m = n/2

        x = np.arange(0, m)/(n-1)

        w = 0.54 - 0.46*np.cos(2 * np.pi * x)

        w_rev = np.flip(w)

        w = np.concatenate((w, w_rev))

    else:

        m = (n+1)/2

        x = np.arange(0, m) / (n - 1)

        w = 0.54 - 0.46 * np.cos(2 * np.pi * x)

        w_rev = np.flip(w[0:len(w)-1])

        w = np.concatenate((w, w_rev))

    return w


def fmel2hz(m):

    return 700*(np.power(10, (m/2595))-1)


def fhz2mel(f):

    return 2595*np.log10(1+f/700)


def buffer(X, n, p=0):

    i = 0

    first_iter = True

    while i < len(X):

        if first_iter:

            result = np.hstack([np.zeros(p), X[:n - p]])

            i = n - p

            # Make 2D array and pivot

            result = np.expand_dims(result, axis=0).T

            first_iter = False

            continue

        # Create next column, add `p` results from last col if given

        col = X[i:i + (n - p)]

        if p != 0:

            col = np.hstack([result[:, -1][-p:], col])

        i += n - p

        # Append zeros if last row and not length `n`

        if len(col) < n:

            col = np.hstack([col, np.zeros(n - len(col))])

        # Combine result with next row

        result = np.hstack([result, np.expand_dims(col, axis=0).T])

    return result


def mfcc(data, rate):

    # MFCC parameters

    wintime = 0.020

    numcep = 13

    lifterexp = 0.6

    minfreq = 0

    maxfreq = 8000

    nbands = 20

    # Pre-compute data

    winpts = np.round(wintime*rate)

    nfft = np.power(2, np.ceil(np.log2(winpts)))

    nfreqs = (int)(nfft/2)+1

    wndw = np.hamming(nfft)

    nframes = np.ceil(data.shape[0]/nfft)

    melwts = np.zeros(((int)(nbands), (int)(nfft)))

    fftfrqs = np.arange(0, nfft)/nfft*rate

    # Get center of mel bands uniformly spaced over given frequency range

    minmel = fhz2mel(minfreq)

    maxmel = fhz2mel(maxfreq)

    binfrqs = fmel2hz(minmel+np.arange(0, (nbands+2)) /
                      (nbands+1)*(maxmel-minmel))

    for i in range(0, nbands):

        fs = [binfrqs[i], binfrqs[i+1], binfrqs[i+2]]

        loslope = (fftfrqs - fs[0])/(fs[1]-fs[0])

        hislope = (fs[2] - fftfrqs)/(fs[2]-fs[1])

        melwts[i, :] = np.maximum(0, np.minimum(loslope, hislope))

    melwts = melwts[:, 0:nfreqs]

    dctm = np.zeros(((int)(numcep), (int)(nbands)))

    for i in range(0, numcep):

        dctm[i, :] = np.cos(i*np.arange(1, 2*nbands, 2) /
                            (2*nbands)*np.pi)*np.sqrt(2/nbands)

    dctm[0, :] = dctm[0, :]/np.sqrt(2)

    A1 = np.array([1])

    A2 = np.power(np.arange(1, numcep), lifterexp)

    liftwts = np.concatenate((A1, A2))

    liftermat = np.diag(liftwts)

    # Compute windowed power spectrum for each frame

    sampbuff = buffer(data, (int)(nfft))

    mult = np.tile(wndw, ((int)(nframes), 1)).transpose()

    sampbuff = np.multiply(sampbuff, mult)

    pspectrum = np.power(np.abs(np.fft.fft(sampbuff, axis=0)), 2)

    pspectrum = pspectrum[0:(int)(nfft/2+1), :]

    # Sum over FFT bins to form mel-scale bins

    aspectrum = np.matmul(melwts, pspectrum)

    # Convert to cepstra via DCT

    cepstra = np.matmul(dctm, np.log(aspectrum+0.00001))

    # Apply liftering weights

    cepstra = np.matmul(liftermat, cepstra)

    return cepstra


# author: Dominik Krzeminski (dokato)


# detrended fluctuation analysis

def calc_rms(x, scale):
    """

    windowed Root Mean Square (RMS) with linear detrending.



    Args:

    -----

      *x* : numpy.array

        one dimensional data vector

      *scale* : int

        length of the window in which RMS will be calculaed

    Returns:

    --------

      *rms* : numpy.array

        RMS data in each window with length len(x)//scale

    """

    # making an array with data divided in windows

    shape = (x.shape[0] // scale, scale)

    X = np.lib.stride_tricks.as_strided(x, shape=shape)

    # vector of x-axis points to regression

    scale_ax = np.arange(scale)

    rms = np.zeros(X.shape[0])

    for e, xcut in enumerate(X):

        coeff = np.polyfit(scale_ax, xcut, 1)

        xfit = np.polyval(coeff, scale_ax)

        # detrending and computing RMS of each window

        rms[e] = np.sqrt(np.mean((xcut - xfit) ** 2))

    return rms


def dfa(x, scale_lim=[5, 9], scale_dens=0.25, show=False):
    """

    Detrended Fluctuation Analysis - measures power law scaling coefficient

    of the given signal *x*.

    More details about the algorithm you can find e.g. here:

    Hardstone, R. et al. Detrended fluctuation analysis: A scale-free

    view on neuronal oscillations, (2012).

    Args:

    -----

      *x* : numpy.array

        one dimensional data vector

      *scale_lim* = [5,9] : list of length 2

        boundaries of the scale, where scale means windows among which RMS

        is calculated. Numbers from list are exponents of 2 to the power

        of X, eg. [5,9] is in fact [2**5, 2**9].

        You can think of it that if your signal is sampled with F_s = 128 Hz,

        then the lowest considered scale would be 2**5/128 = 32/128 = 0.25,

        so 250 ms.

      *scale_dens* = 0.25 : float

        density of scale divisions, eg. for 0.25 we get 2**[5, 5.25, 5.5, ... ]

      *show* = False

        if True it shows matplotlib log-log plot.

    Returns:

    --------

      *scales* : numpy.array

        vector of scales (x axis)

      *fluct* : numpy.array

        fluctuation function values (y axis)

      *alpha* : float

        estimation of DFA exponent

    """

    # cumulative sum of data with substracted offset

    y = np.cumsum(x - np.mean(x))

    scales = (2 ** np.arange(scale_lim[0],
              scale_lim[1], scale_dens)).astype(int)

    fluct = np.zeros(len(scales))

    # computing RMS for each window

    for e, sc in enumerate(scales):

        fluct[e] = np.sqrt(np.mean(calc_rms(y, sc) ** 2))

    # fitting a line to rms data

    coeff = np.polyfit(np.log2(scales), np.log2(fluct), 1)

    if show:

        fluctfit = 2 ** np.polyval(coeff, np.log2(scales))

        plt.loglog(scales, fluct, 'bo')

        plt.loglog(scales, fluctfit, 'r', label=r'$\alpha$ = %0.2f' % coeff[0])

        plt.title('DFA')

        plt.xlabel(r'$\log_{10}$(time window)')

        plt.ylabel(r'$\log_{10}$<F(t)>')

        plt.legend()

        plt.show()

    return scales, fluct, coeff[0]
