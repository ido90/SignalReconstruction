# Reconstruction of signals with dropped samples: methods comparison

## Introduction
A signal is available in discrete times with uniform gaps, up to certain missing points (*dropped samples*).
Fourier transform assumes uniform grid of points, and tends to be sensitive to such dropped samples.
The goal of this project is to compare - in both signal space and Fourier space - the errors of the reconstructed signal using various reconstruction methods.

## Data
### Signals
- **single_freq**: a single frequency (sinusoidal).
- **ten_freqs**: a sum of ten frequencies (ten sinusoidals).
- **rand_gauss**: random iid Gaussian samples.
- **earthquake**: acoustic signal before an earthquake.

### Masks
- **structured**: 48 dropped samples every 4096 samples.
- **random**: 3% randomly dropped samples.
- **rand_seqs**: random sequences of dropped samples.

### Methods
- **ignore**: just apply fft on the corrupted signal.
- **zeros**: replace missing samples with zeros.
- **linear**: linear interpolation.
- **sp_x**: spline of order x.
- [**nfft**](https://github.com/jakevdp/nfft/blob/master/README.md): omitted due to poor results.

## Results
- **Ignoring the missing samples yields terrible Fourier transform**.
- **Linear interpolation and linear spline look very good in all metrics**.
- High-order splines (quadratic/cubic) tend to yield large deviations in presence of long gaps of missing samples, causing large RMSE in signal space and large maximum error in Fourier space, although they sometimes have good RMSE in Fourier space.
- Naively filling the gaps with zeros is not very good due to arbitrary jumps, yet by comparison to the ignoring method it seems to solve most of the problem.

|![](https://github.com/ido90/SignalReconstruction/blob/master/Output/masked_signals_2.png)|
|:--:|
| Signals & masks |

|![](https://github.com/ido90/SignalReconstruction/blob/master/Output/reconstructed_signals.png)|
|:--:|
| Reconstructed signals (RMSE in paranthesis) |

|![](https://github.com/ido90/SignalReconstruction/blob/master/Output/reconstructed_ffts_focused.png)|
|:--:|
| A sample of the reconstructed FFTs (RMSE in paranthesis) |

|![](https://github.com/ido90/SignalReconstruction/blob/master/Output/reconstruction_errors.png)|
|:--:|
| Summary of errors |

_________________

## References

Most references recommend to deal with missing data points in spectral analysis by filling in the missing points using some [interpolation](http://mres.uni-potsdam.de/index.php/2017/08/22/data-voids-and-spectral-analysis-dont-be-afraid-of-gaps/) (linear or spline, typically for uniform sampling with missing points) or [Gaussian filter](https://scicomp.stackexchange.com/questions/593/how-do-i-take-the-fft-of-unevenly-spaced-data) (typically for non-uniform sampling).

[Some references](https://dsp.stackexchange.com/questions/22930/spectral-analysis-of-a-time-series-with-missing-data-points) suggest applying FFT for each uniform interval separately, and averaging the results, which I was too lazy to try.
[This one](https://electronics.stackexchange.com/questions/290994/fast-fourier-transformation-of-incomplete-signals) suggests weighted Fourier transform, for which I didn't easily find an available implementation; and non-equispaced-FFT (NFFT), for which I used this [implementation](https://github.com/jakevdp/nfft/) which I didn't find stable enough.
