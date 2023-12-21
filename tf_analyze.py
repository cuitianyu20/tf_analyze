#!/usr/bin/env python
# -*- coding: utf-8 -*-
import fcwt
import matplotlib
import numpy as np
from obspy.signal import util
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import NullFormatter

def fcwt_func(st, sampling_rate, fmin, fmax, fn, nthreads=8, logspace=True, 
              use_optimization_plan=False, use_normalization=True):
    """
    The fast Continuous Wavelet Transformation in the Frequency Domain (fCWT).
    .. see article::[Arts et al., 2022]
    :param st: time dependent signal.
    :param sampling_rate: sampling rate of the signal
    :param fmin: minimum frequency (in Hz)
    :param fmax: maximum frequency (in Hz)
    :param fn: number of logarithmically or linearly spaced frequencies between fmin and fmax
    :param nthreads: number of threads to use for the calculation
    :param logspace: if True, logarithmically spaced frequencies are used, otherwise linearly spaced frequencies are used
    :param use_optimization_plan: if True, an optimization plan is used to speed up the calculation
    :param use_normalization: if True, the output is normalized
    :return: time frequency representation of st, type numpy.ndarray of complex values, shape = (fn, len(st)).
    """
    #initialize Morlet wavelet with wavelet parameter (sigma) 2.0
    morl = fcwt.Morlet(2.0)
    #initialize scales
    if logspace:
        scales = fcwt.Scales(morl, fcwt.FCWT_LOGSCALES, fs=int(sampling_rate), f0=fmin, f1=fmax, fn=fn)
    else:
        scales = fcwt.Scales(morl, fcwt.FCWT_LINFREQS, fs=int(sampling_rate), f0=fmin, f1=fmax, fn=fn)
    #initialize fcwt
    fcwt_obj = fcwt.FCWT(morl, nthreads, use_optimization_plan, use_normalization)   
    #initialize output array
    cwt_out = np.zeros((fn, st.shape[-1]), dtype=np.complex64)
    #calculate cwt
    st = st.astype('float32')
    fcwt_obj.cwt(st, scales, cwt_out)
    return np.flip(cwt_out,axis=0)


def plot_tfr_func(data, data_label, dt_list, twin_min=-10, twin_max=10, fmin=0.01, fmax=10, nf=200, spec_major_locator=2, spec_minor_locator=1,
                     time_major_locator=5, time_minor_locator=2.5, w_spec=0.5, h_time=0.6, w_bar=0.15, space_bar=0.35, line_width=1.5, font_size=10,
                     line_color=['black', '#D70200',], cmap='hot_r', fft_zero_pad_fac=0, mode='absolute', normalize=True, log=False, 
                     mixed_cols=None, savename='tfr', figsize=(10, 8)): 
    """
    Plot the time-frequency representation of a signal.
    :param data: time dependent signal, type numpy.ndarray, shape = (npts,)
    :param data_label: labels of the data
    :param dt_list: sampling interval of the signals
    :param twin_min: minimum time window (in seconds)
    :param twin_max: maximum time window (in seconds)
    :param fmin: minimum frequency (in Hz)
    :param fmax: maximum frequency (in Hz)
    :param nf: number of logarithmically or linearly spaced frequencies between fmin and fmax
    :param spec_major_locator: major locator of the frequency spectrum
    :param spec_minor_locator: minor locator of the frequency spectrum
    :param time_major_locator: major locator of the time signal
    :param time_minor_locator: minor locator of the time signal
    :param w_spec: width of the frequency spectrum
    :param h_time: height of the time signal
    :param w_bar: width of the colorbar
    :param space_bar: space between the colorbar and the time-frequency data
    :param line_width: linewidth of the time signal and the frequency spectrum
    :param font_size: font size of the figure labels
    :param line_color: color of the time signal and the frequency spectrum
    :param cmap: colormap to use
    :param fft_zero_pad_fac: integer, if > 0, the signal is zero padded to
        ``nfft = next_pow_2(len(st)) * fft_zero_pad_fac`` to get smoother
        spectrum in the low frequencies (has no effect on the TFR and might
        make demeaning/tapering necessary to avoid artifacts)
    :param mode: 'absolute' for absolute value of TFR, 'power' for ``|TFR|^2``   
    :param normalize: if True, the output is normalized
    :param log: if True, the frequencies are plotted on a logarithmic scale
    :param mixed_cols: if True, plot the time-frequency representation in a mixed col
    :param savename: name or path of the figure
    :param figsize: size of the figure
    :author: Tianyu Cui
    :date: 2023-12-20
    :return: figure
    """
    def smooth_spectrum(fft_zero_pad_fac, npts):
        # smooth spectrum in low frequencies
        if fft_zero_pad_fac == 0:
            nfft = npts
        else:
            nfft = util.next_pow_2(npts) * fft_zero_pad_fac
        return nfft
    if not isinstance(data, list):
        raise TypeError('Input data must be a list')
    if not isinstance(data_label, list):
        raise TypeError('Input data labels must be a list')
    ntr = len(data)
    npts_list = [data[i].shape[-1] for i in range(ntr)]
    t_win_list = [np.linspace(twin_min, twin_max, npts_list[i]) for i in range(ntr)]
    Ny_list = [1 / (2 * dt_list[i]) for i in range(ntr)] # Nyquist frequency
    nfft_list = [smooth_spectrum(fft_zero_pad_fac, npts_list[i]) for i in range(ntr)]
    freq_list = [np.linspace(0, Ny_list[i], nfft_list[i] // 2 + 1) for i in range(ntr)]
    _tfr = [np.zeros((ntr, nf, npts), dtype=np.complex64) for npts in npts_list]
    _spec = [np.zeros((ntr, nfft // 2 + 1), dtype=np.complex64) for nfft in nfft_list]
    for i in range(ntr):
        _tfr[i] = fcwt_func(data[i], 1/dt_list[i], fmin, fmax, nf, nthreads=8, logspace=log,
                            use_optimization_plan=False, use_normalization=True)
        _spec[i] = np.fft.rfft(data[i], n=nfft_list[i])
    if mode == 'absolute':
        for i in range(ntr):
            _tfr[i] = np.abs(_tfr[i])
            _spec[i] = np.abs(_spec[i])
    elif mode == 'power':
        for i in range(ntr):
            _tfr[i] = np.abs(_tfr[i]) ** 2
            _spec[i] = np.abs(_spec[i]) ** 2
    else:
        raise ValueError('mode "' + mode + '" not defined!')
    # Normalize
    if normalize:
        for i in range(ntr):
            _tfr[i] /= np.max(_tfr[i])
            _spec[i] /= (len(data[i])/2)
    # plot the time-frequency representation
    def devide_figs(rows, cols, left_ini, bottom_ini, width_ini, height_ini, coor_x_len, coor_y_len):
        subplots = []
        for i in range(rows):
            for j in range(cols):
                rect = [left_ini + j * coor_x_len,
                        bottom_ini - i * coor_y_len,
                        width_ini,
                        height_ini]
                subplots.append(fig.add_axes(rect))
        return subplots

    fig = plt.figure(figsize=figsize)
    # rows, cols and divide the figure for time and frequency data
    if mixed_cols:
        cols = mixed_cols
        rows = ntr // cols
    else:
        if ntr == 1:
            rows = 1
            cols = 1
        else:
            rows = int(np.ceil(np.sqrt(ntr)))
            cols = ntr // rows + 1
    divide_num_x = 4
    divide_num_y = 4
    coor_x_mini = 1/(cols*divide_num_x)
    coor_y_mini = 1/(rows*divide_num_y)
    coor_x_len = 1/cols
    coor_y_len = 1/rows
    left = coor_x_mini
    bottom = coor_y_mini + coor_y_len * (rows-1)
    width = coor_x_mini * divide_num_x / 2
    height = coor_y_mini * divide_num_y / 2
    # frequency spectrum data and time data
    width_spec = coor_x_mini * w_spec
    height_time = coor_y_mini * h_time
    left_spec = left - width_spec
    bottom_time = bottom - height_time
    # colorbar
    left_colorbar = left + width + coor_x_mini * space_bar
    width_colorbar = coor_x_mini * w_bar
    # set linewidth
    plt.rcParams['axes.linewidth'] = 1.0
    matplotlib.rcParams['font.size'] = font_size
    # time and frequency data
    subplots_tfr = devide_figs(rows, cols, left, bottom, width, height, coor_x_len, coor_y_len)
    subplots_time = devide_figs(rows, cols, left, bottom_time, width, height_time, coor_x_len, coor_y_len)
    subplots_spec = devide_figs(rows, cols, left_spec, bottom, width_spec, height, coor_x_len, coor_y_len)
    subplots_colorbar = devide_figs(rows, cols, left_colorbar, bottom, width_colorbar, height, coor_x_len, coor_y_len)
    # plot the time-frequency data
    for i, ax_tfr in enumerate(subplots_tfr):
        if i < ntr:
            if log:
                x, y = np.meshgrid(t_win_list[i], np.logspace(np.log10(fmin), np.log10(fmax), _tfr[i].shape[0]))
                ax_tfr.set_ylabel('Frequency (Hz)')
                ax_tfr.yaxis.set_label_position('right')
                ax_tfr.set_yscale('log')
            else:
                x, y = np.meshgrid(t_win_list[i], np.linspace(fmin, fmax, _tfr[i].shape[0]))
                # add axis ticks
                ax_tfr.yaxis.set_major_locator(ticker.MultipleLocator(spec_major_locator))
                ax_tfr.yaxis.set_minor_locator(ticker.MultipleLocator(spec_minor_locator))
            img_tfr = ax_tfr.pcolormesh(x, y, _tfr[i], cmap=cmap, shading='nearest')
            img_tfr.set_rasterized(True)
            ax_tfr.set_xlim(twin_min, twin_max)
            ax_tfr.set_ylim(fmin, fmax)
            ax_tfr.xaxis.set_major_locator(ticker.MultipleLocator(time_major_locator))
            ax_tfr.xaxis.set_minor_locator(ticker.MultipleLocator(time_minor_locator))
            # plot grid, mode both: major and minor grid lines are shown
            ax_tfr.grid(True, which='major', linestyle='--', linewidth=1.0, alpha=0.7)
            # remove axis labels
            ax_tfr.xaxis.set_major_formatter(NullFormatter())
            ax_tfr.yaxis.set_major_formatter(NullFormatter())
            ax_tfr.set_title('%s' % data_label[i])
        else:
            ax_tfr.set_visible(False)
    # plot time signals data
    for i, ax_sig in enumerate(subplots_time):
        if i < ntr:
            ax_sig.plot(t_win_list[i], data[i], color=line_color[0], linewidth=line_width)
            ax_sig.set_xlim(twin_min, twin_max)
            ax_sig.set_ylim(np.min(data[i])*1.1, np.max(data[i])*1.1)
            ax_sig.set_xlabel('Time (s)')
            ax_sig.yaxis.tick_right()
            ax_sig.xaxis.set_major_locator(ticker.MultipleLocator(time_major_locator))
            ax_sig.xaxis.set_minor_locator(ticker.MultipleLocator(time_minor_locator))
        else:
            ax_sig.set_visible(False)
    # plot the frequency spectrum data
    for i, ax_spec in enumerate(subplots_spec):
        if i < ntr:
            if log:
                ax_spec.semilogy(_spec[i], freq_list[i], color=line_color[1], linewidth=line_width)
                ax_spec.set_yscale('log')
                ax_spec.set_ylim(fmin, fmax)
            else:
                ax_spec.plot(_spec[i], freq_list[i],  color=line_color[1], linewidth=line_width)
                ax_spec.set_ylim(int(fmin), fmax)
                ax_spec.yaxis.set_major_locator(ticker.MultipleLocator(spec_major_locator)) 
                ax_spec.yaxis.set_minor_locator(ticker.MultipleLocator(spec_minor_locator)) 
                ax_spec.set_ylabel('Frequency (Hz)')
            ax_spec.set_xlim(np.max(_spec[i])*1.1, 0)
        else:
            ax_spec.set_visible(False)
    # plot colorbar
    for i, ax_bar in enumerate(subplots_colorbar):
        if (i < ntr) and ((i+1) % cols == 0):
            tfr_min = int(np.min(_tfr[i]))
            tfr_max = np.max(_tfr[i])
            locator = np.linspace(tfr_min, tfr_max, 5)
            colorbar = fig.colorbar(img_tfr, ax_bar, orientation='vertical')
            tick_locator = ticker.MaxNLocator(nbins=5)
            colorbar.locator = tick_locator
            colorbar.set_ticks(locator)
            colorbar.update_ticks()
            colorbar.ax.set_title('Power', pad=10)
        else:
            ax_bar.set_visible(False)
    # save figure
    plt.savefig('%s.png' % savename, dpi=300)



if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
