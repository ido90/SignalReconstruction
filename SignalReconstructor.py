import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pandas as pd
from scipy import interpolate
from nfft import nfft
import re
from time import time
from pprint import pprint
import os
from warnings import warn
import utils
from InteractivePlotter import InteractiveFigure

COLORS = ('blue','red','green','purple','orange','grey','pink','lime')

class SignalReconstructor:

    def __init__(self, n_samples=150e3, signals=None, masks=None, methods=None,
                 detailed=0):
        self.n_samples = int(n_samples)
        self.t = np.arange(self.n_samples)
        self.signals = {}
        self.masks = {}
        self.methods = {}

        if signals is None:
            self.add_default_signals(detailed=detailed)
        else:
            self.signals = signals
        if masks is None:
            self.add_default_masks(detailed=detailed)
        else:
            self.masks = masks
        if methods is None:
            self.add_default_methods(detailed=detailed)
        else:
            self.methods = methods

    def add_default_signals(self, detailed=0):
        n = self.n_samples
        T = np.random.exponential(20, 10)
        self.signals['single_freq'] = np.sin(1/T[0]*np.arange(n))
        if detailed >= 1:
            self.signals['ten_freqs'] = np.zeros(n)
            for t in T:
                self.signals['ten_freqs'] += np.sin(1/t*np.arange(n))
        if detailed >= 2:
            self.signals['white'] = np.fft.ifft(np.ones(n))
            self.signals['rand_unif'] = np.random.uniform(0,1,n)
        if detailed >= 1:
            self.signals['rand_gauss'] = np.random.normal(0,1,n)
        if os.path.isfile('Data/signal.csv'):
            self.signals['earthquake'] = \
                pd.read_csv('Data/signal.csv', nrows=n).iloc[:,1]

    def add_default_masks(self, detailed=0):
        n = self.n_samples
        self.masks['structured'] = np.zeros(n, dtype=bool)
        for i in range(int(n/4096)):
            self.masks['structured'][((i+1)*4096-48+1):((i+1)*4096+1)] = True
        if detailed >= 1:
            self.masks['random'] = np.random.choice((True,False), n, p=(0.03,0.97))
        self.masks['rand_seqs'] = np.zeros(n, dtype=bool)
        i = 0
        val = False
        while i<n:
            next_i = i + round(np.random.exponential(20 if val else 3e2))
            self.masks['rand_seqs'][i:(next_i+1)] = val
            i = next_i
            val = not val
        # make sure edges are available
        for mask in self.masks:
            self.masks[mask][ 0] = False
            self.masks[mask][-1] = False

    def add_default_methods(self, detailed=0):
        self.methods['ignore'] = lambda t,x: rem_nans(t,x)
        if detailed >= 1:
            self.methods['zeros'] = lambda t,x: fill_zeros(t,x)
        self.methods['linear'] = lambda t,x: interpolation(t, x, 'linear')
        if detailed >= 2:
            self.methods['sp_0'] = lambda t,x: interpolation(t, x, 'zero')
            self.methods['sp_1'] = lambda t,x: interpolation(t, x, 'slinear')
        self.methods['sp_2'] = lambda t,x: interpolation(t, x, 'quadratic')
        self.methods['sp_3'] = lambda t,x: interpolation(t, x, 'cubic')
        self.methods['averaged'] = lambda t,x: averaged_fft(t, x)
        if detailed >= 3:
            self.methods['nfft'] = lambda t,x: apply_nfft(t,x)

    def plot_masked_signals(self, n_max=None):
        n = len(self.t[:n_max])
        fig, axs = plt.subplots(len(self.masks), len(self.signals))
        for i, mask in enumerate(self.masks):
            for j, sig in enumerate(self.signals):
                ax = axs[i,j]
                ids = np.logical_not(self.masks[mask])[:n_max]
                ax.plot(self.t[:n_max][ids], self.signals[sig][:n_max][ids],
                        'b.' if n<1e3 else 'b,')
                ids = self.masks[mask][:n_max]
                ax.plot(self.t[:n_max][ids], self.signals[sig][:n_max][ids], 'r.')
                ax.grid()
                if j==0: ax.set_ylabel('Mask: ' + mask, fontsize=12)
                if i==0: ax.set_title('Signal: ' + sig, fontsize=12)
                utils.draw()

    def ruined_signal(self, signal, mask):
        s = self.signals[signal].copy()
        s[self.masks[mask]] = np.nan
        return s

    def plot_reconstructions(self, n_max=None, plot_missing=False):
        tp1 = self.t[4000:4000 + n_max] if n_max else self.t
        tp2 = self.t[:n_max] if n_max else self.t
        fig1, axs1 = plt.subplots(len(self.masks), len(self.signals))
        fig2, axs2 = plt.subplots(len(self.masks), len(self.signals))
        rmse = {m: {} for m in self.methods}
        frmse = {m: {} for m in self.methods}
        f_max_err = {m: {} for m in self.methods}
        for i, mask in enumerate(self.masks):
            for j, sig in enumerate(self.signals):
                ax1 = axs1[i,j]
                #ax1 = InteractiveFigure()
                ax2 = axs2[i,j]
                good_sig = self.signals[sig]
                good_fft = np.fft.fft(good_sig, norm='ortho')
                good_fft /= np.mean(np.abs(good_fft))
                bad_sig = self.ruined_signal(sig, mask)
                xp = good_sig[4000:4000 + n_max] if n_max else good_sig
                ax1.plot(tp1, xp, color='black', linewidth=0.8, label='original')
                xp = [np.mean(np.abs(good_fft)[int(i*len(good_fft)/n_max):int((i+1)*len(good_fft)/n_max)])
                      for i in range(n_max)] \
                    if n_max else np.abs(good_fft)
                ax2.plot(tp2[1:-1], xp[1:-1], color='black', linewidth=0.8, label='original')
                for k,m in enumerate(self.methods):
                    # reconstruct
                    t, x, f = self.methods[m](self.t, self.ruined_signal(sig, mask))
                    x_long = np.concatenate((x, np.zeros(len(good_sig)-len(x))))
                    f_long = np.concatenate((f, np.zeros(len(good_fft)-len(f))))
                    f_long /= np.mean(np.abs(f_long))
                    # mse
                    err = np.sqrt(np.mean((x_long-good_sig)**2))
                    rmse[m][mask+' :\n'+sig] = err
                    xp = x_long[4000:4000+n_max] if n_max else x_long
                    if plot_missing or len(x) == len(good_sig):
                        plt.figure(fig1.number)
                        ax1.plot(tp1, xp, color=COLORS[k], linewidth=0.8, label=m+f' ({err:.1f})')
                    # fourier mse
                    err = np.sqrt(np.mean((np.abs(f_long)-np.abs(good_fft))**2))
                    frmse[m][mask+' :\n'+sig] = err
                    f_max_err[m][mask+' :\n'+sig] = np.sqrt(np.max((np.abs(f_long[1:-1])-np.abs(good_fft[1:-1]))**2))
                    xp = [np.mean(np.abs(f_long)[int(i*len(f_long)/n_max):int((i+1)*len(f_long)/n_max)])
                      for i in range(n_max)] \
                        if n_max else np.abs(f_long)
                    if plot_missing or len(f) == len(good_fft):
                        plt.figure(fig2.number)
                        ax2.plot(tp2[1:-1], xp[1:-1], color=COLORS[k], linewidth=0.8, label=m + f' ({err:.1f})')
                for fig,ax in zip((fig1,fig2),(ax1,ax2)):
                    plt.figure(fig.number)
                    ax.grid()
                    ax.legend()
                    if j==0: ax.set_ylabel('Mask: ' + mask, fontsize=12)
                    if i==0: ax.set_title('Signal: ' + sig, fontsize=12)
            plt.figure(fig1.number)
            utils.draw()
            plt.figure(fig2.number)
            utils.draw()
        # plot all RMSEs
        fig, axs = plt.subplots(3, 1)
        self.plot_rmse(rmse,  axs[0], title='Signal (RMSE)', logy=True)
        self.plot_rmse(frmse, axs[1], title='Fourier (RMSE of normalized signal)', logy=True)
        self.plot_rmse(f_max_err, axs[2], title='Fourier (Max Error of normalized signal)', logy=True)

    def plot_rmse(self, rmse, ax, title='', logy=False):
        M = np.max([v for r in rmse.values() for v in r.values()])
        keys = list(rmse[list(self.methods.keys())[0]].keys())
        methods = list(self.methods.keys())
        width = 1 / (len(methods) + 1)
        for i,m in enumerate(methods):
            rects = ax.bar(
                np.arange(len(keys)) + i * width,
                [rmse[m][k] for k in keys],
                width
            )
            for rect, k in zip(rects, keys):
                height = rect.get_height()
                ax.text(rect.get_x() + rect.get_width() / 2., height + 0.1*M,
                        f'{m:s}',
                        ha='center', va='bottom', rotation=90)
        ax.grid(axis='y')
        if logy: ax.set_yscale('log')
        if title: ax.set_title(title, fontsize=14)
        ax.set_ylabel('error', fontsize=12)
        ax.set_xlabel('mask : signal', fontsize=12)
        ax.set_xticks(np.arange(len(keys)) + len(methods) * width / 2)
        ax.set_xticklabels(keys)
        utils.draw()


def interpolation(t, x, kind):
    ids = np.logical_not(np.isnan(x))
    p = interpolate.interp1d(t[ids], x[ids], kind=kind)
    x = p(t)
    return (t, x, np.fft.fft(x, norm='ortho'))

def rem_nans(t, x):
    ids = np.logical_not(np.isnan(x))
    return (t[ids], x[ids], np.fft.fft(x[ids], norm='ortho'))

def fill_zeros(t, x):
    x[np.isnan(x)] = 0
    return (t, x, np.fft.fft(x, norm='ortho'))

def apply_nfft(t, x):
    ids = np.logical_not(np.isnan(x))
    if np.sum(ids)%2 != 0:
        ids[np.where(ids)[-1]] = False
    try:
        f = nfft(t[ids], x[ids])
    except:
        f = np.fft.fft(x[ids], norm='ortho')
    return(t[ids], x[ids], f)

def averaged_fft(t, x, min_len=100):
    # find sequences
    interval_starts = np.where(np.logical_and(np.isnan(x[:-1]),np.logical_not(np.isnan(x[1:]))))[0] + 1
    interval_ends = np.where(np.logical_and(np.logical_not(np.isnan(x[:-1])),np.isnan(x[1:])))[0] + 1
    if not np.isnan(x[0]):
        interval_starts = np.concatenate(([0], interval_starts))
    if not np.isnan(np.array(x)[-1]):
        interval_ends = np.concatenate((interval_ends, [len(x)]))

    # compute ffts
    # n = np.min([tf-ti for ti, tf in zip(interval_starts, interval_ends)
    #             if tf-ti>=min_len])
    n = len(x)
    ffts = []
    for ti, tf in zip(interval_starts, interval_ends):
        if tf-ti<min_len:
            continue
        f = np.fft.fft(x[ti:tf], norm='ortho')
        p = interpolate.interp1d(np.arange(len(f))/(1*(len(f)-1)), f, kind='linear')
        f = p(np.arange(n)/(1*n))
        ffts.append(np.abs(f))

    # average ffts
    avg_fft = np.mean(ffts, axis=0)

    ids = np.logical_not(np.isnan(x))
    return (t[ids], x[ids], avg_fft)


if __name__=='__main__':
    x = SignalReconstructor(n_samples=150e3)
    x.plot_masked_signals(n_max=256)
    x.plot_reconstructions(n_max=256)
    plt.show()
