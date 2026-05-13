"""
Capture file diagnostic plots.

Produces a single PNG with three panels:
  - Power spectral density (Welch method)
  - Time domain snippet — I and Q channels
  - Sample amplitude histogram — I and Q channels
"""

import matplotlib
matplotlib.use('Agg')  # non-interactive backend; must precede pyplot import

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from scipy.signal import welch

from gnss_common.capture.capture_file import CaptureFile
from gnss_common.capture.capture_metadata import CaptureMetadata


def inspect_capture(
    capture: CaptureFile,
    metadata: CaptureMetadata,
    output_path: Path | str,
    num_ms: float = 5.0,
    skip_ms: float = 0.0,
) -> None:
    """
    Write a diagnostic PNG for a capture file.

    Parameters
    ----------
    capture : CaptureFile
        Open capture file to read from.
    metadata : CaptureMetadata
        Signal parameters: sample_rate, center_freq, is_complex.
    output_path : Path or str
        Destination path for the output PNG.
    num_ms : float
        Duration of data to read, in milliseconds.  Default 5 ms.
    skip_ms : float
        How far into the file to begin reading, in milliseconds.  Default 0.
    """
    output_path = Path(output_path)

    # ------------------------------------------------------------------
    # Read data
    # ------------------------------------------------------------------
    start_sample = metadata.ms_to_samples(skip_ms)
    num_samples  = metadata.ms_to_samples(num_ms)
    samples = capture.read(start=start_sample, count=num_samples)

    if metadata.is_complex:
        i_data = np.real(samples)
        q_data = np.imag(samples)
    else:
        i_data = samples
        q_data = None

    t_ms = np.arange(num_samples) / metadata.sample_rate * 1e3  # time axis in ms

    # ------------------------------------------------------------------
    # Figure layout: PSD full-width on top, time and histogram below
    # ------------------------------------------------------------------
    fig = plt.figure(figsize=(12, 9))
    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.50, wspace=0.35)

    ax_psd    = fig.add_subplot(gs[0, :])
    ax_time_i = fig.add_subplot(gs[1, 0])
    ax_time_q = fig.add_subplot(gs[1, 1])
    ax_hist_i = fig.add_subplot(gs[2, 0])
    ax_hist_q = fig.add_subplot(gs[2, 1])

    # ------------------------------------------------------------------
    # Power spectral density (Welch)
    # nperseg scaled to data length; 32768 is a reasonable ceiling for
    # 5 ms at 18 Msps (90 000 samples)
    # ------------------------------------------------------------------
    nperseg = min(32768, num_samples // 4)

    if metadata.is_complex:
        f, pxx = welch(
            samples,
            fs=metadata.sample_rate,
            nperseg=nperseg,
            return_onesided=False,
        )
        # fftshift re-centres the two-sided spectrum so DC is in the middle;
        # add center_freq to place the x-axis in receiver IF coordinates
        f   = np.fft.fftshift(f) + metadata.center_freq
        pxx = np.fft.fftshift(pxx)
    else:
        f, pxx = welch(samples, fs=metadata.sample_rate, nperseg=nperseg)
        f = f + metadata.center_freq

    ax_psd.plot(f / 1e6, 10 * np.log10(pxx), linewidth=0.8)
    ax_psd.set_xlabel('Frequency (MHz)')
    ax_psd.set_ylabel('PSD (dB/Hz)')
    ax_psd.set_title('Power Spectral Density')
    ax_psd.grid(True)

    # ------------------------------------------------------------------
    # Time domain — show a 0.5 ms snippet to keep the plot readable
    # ------------------------------------------------------------------
    plot_samples = min(num_samples, metadata.ms_to_samples(0.5))

    ax_time_i.plot(t_ms[:plot_samples], i_data[:plot_samples], linewidth=0.5)
    ax_time_i.set_xlabel('Time (ms)')
    ax_time_i.set_ylabel('Amplitude')
    ax_time_i.set_title('Time Domain — I')
    ax_time_i.grid(True)

    if q_data is not None:
        ax_time_q.plot(t_ms[:plot_samples], q_data[:plot_samples],
                       linewidth=0.5, color='C1')
        ax_time_q.set_xlabel('Time (ms)')
        ax_time_q.set_ylabel('Amplitude')
        ax_time_q.set_title('Time Domain — Q')
        ax_time_q.grid(True)
    else:
        ax_time_q.set_visible(False)
        ax_hist_q.set_visible(False)

    # ------------------------------------------------------------------
    # Histograms
    # Bins are integer-centred for int8 data; works for float data too
    # ------------------------------------------------------------------
    bins = np.arange(-129, 130) + 0.5  # edges between integers

    ax_hist_i.hist(i_data, bins=bins, color='C0')
    ax_hist_i.set_xlabel('Sample value')
    ax_hist_i.set_ylabel('Count')
    ax_hist_i.set_title('Histogram — I')
    ax_hist_i.grid(True, axis='y')

    if q_data is not None:
        ax_hist_q.hist(q_data, bins=bins, color='C1')
        ax_hist_q.set_xlabel('Sample value')
        ax_hist_q.set_ylabel('Count')
        ax_hist_q.set_title('Histogram — Q')
        ax_hist_q.grid(True, axis='y')

    # ------------------------------------------------------------------
    # Overall title and save
    # ------------------------------------------------------------------
    fig.suptitle(
        f'fs = {metadata.sample_rate / 1e6:.3f} MHz   '
        f'IF = {metadata.center_freq / 1e3:.1f} kHz   '
        f'({num_ms:.0f} ms read from offset {skip_ms:.0f} ms)',
        fontsize=11,
    )

    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)