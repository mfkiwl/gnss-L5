"""
check_capture.py — diagnostic script for the L5 capture file.

Reads 10 ms of data and reports:
  - File size, sample count, and implied duration
  - I and Q value ranges, means, and standard deviations
  - Histogram of I and Q sample values
  - Power spectral density (Welch, 1 ms segments)

Run from the project root:
    python scripts/check_capture.py

Output images are written to the current working directory.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless — no display required (WSL-safe)
import matplotlib.pyplot as plt
from scipy.signal import welch

from gnss_common.capture.capture_file import CaptureFile
from gnss_common.capture.capture_metadata import CaptureMetadata
from gnss_l5.constants import L5_NOMINAL_FREQ_HZ

# ---------------------------------------------------------------------------
# Capture-specific constants — edit to match the file under test
# ---------------------------------------------------------------------------

CAPTURE_PATH:    Path  = Path("/mnt/e/L5_capture/L5_IF20KHz_FS18MHz")
SAMPLE_RATE:     float = 18.0e6    # Hz
RESIDUAL_IF_HZ:  float = 20.0e3   # Hz — hardware tuning offset
CENTER_FREQ:     float = L5_NOMINAL_FREQ_HZ - RESIDUAL_IF_HZ

ANALYSIS_MS:     int   = 10        # milliseconds to read and analyse


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _print_section(title: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:

    # ------------------------------------------------------------------
    # Open file
    # ------------------------------------------------------------------
    if not CAPTURE_PATH.exists():
        print(f"ERROR: capture file not found: {CAPTURE_PATH}", file=sys.stderr)
        sys.exit(1)

    cf = CaptureFile(CAPTURE_PATH, dtype=np.int8, is_complex=True)
    metadata = CaptureMetadata(
        sample_rate=SAMPLE_RATE,
        center_freq=CENTER_FREQ,
        dtype=np.int8,
        is_complex=True,
    )

    file_bytes   = CAPTURE_PATH.stat().st_size
    file_dur_s   = cf.num_samples / SAMPLE_RATE

    _print_section("File summary")
    print(f"  Path          : {CAPTURE_PATH}")
    print(f"  Size          : {file_bytes / 1e9:.3f} GB  ({file_bytes:,} bytes)")
    print(f"  Total samples : {cf.num_samples:,}")
    print(f"  Duration      : {file_dur_s:.2f} s  ({file_dur_s * 1e3:.0f} ms)")
    print(f"  Sample rate   : {SAMPLE_RATE / 1e6:.1f} MHz")
    print(f"  Residual IF   : {RESIDUAL_IF_HZ / 1e3:.1f} kHz")

    # ------------------------------------------------------------------
    # Read data
    # ------------------------------------------------------------------
    n_samples = metadata.ms_to_samples(ANALYSIS_MS)
    _print_section(f"Reading {ANALYSIS_MS} ms  ({n_samples:,} samples)")
    samples = cf.read(count=n_samples)
    print(f"  Output dtype  : {samples.dtype}")
    print(f"  Output shape  : {samples.shape}")

    i_vals = samples.real       # float64, but values are integer-valued (int8 origin)
    q_vals = samples.imag

    # ------------------------------------------------------------------
    # Per-channel statistics
    # ------------------------------------------------------------------
    _print_section("Channel statistics")
    for name, vals in (("I", i_vals), ("Q", q_vals)):
        print(
            f"  {name}  min={int(vals.min()):+5d}  max={int(vals.max()):+5d}"
            f"  mean={vals.mean():+7.3f}  std={vals.std():6.3f}"
        )

    mean_power = (np.abs(samples) ** 2).mean()
    print(f"\n  Mean power    : {mean_power:.3f}  ({10 * np.log10(mean_power):.2f} dBFS)")

    # ------------------------------------------------------------------
    # Histogram of I and Q values
    # ------------------------------------------------------------------
    # int8 values run from -128 to +127; one bin per integer value.
    bins = np.arange(-128.5, 128.5, 1.0)   # 256 unit-width bins

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=False)
    fig.suptitle(
        f"I/Q value histograms — {ANALYSIS_MS} ms of {CAPTURE_PATH.name}",
        fontsize=11,
    )

    for ax, vals, label, colour in (
        (axes[0], i_vals, "I channel", "steelblue"),
        (axes[1], q_vals, "Q channel", "coral"),
    ):
        ax.hist(vals, bins=bins, color=colour, edgecolor="none")
        ax.set_title(label)
        ax.set_xlabel("Sample value (int8)")
        ax.set_ylabel("Count")
        ax.set_xlim(-130, 130)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    hist_path = Path("check_capture_histogram.png")
    plt.savefig(hist_path, dpi=120)
    plt.close(fig)
    print(f"\n  Histogram saved : {hist_path.resolve()}")

    # ------------------------------------------------------------------
    # Power spectral density (Welch, 1 ms segments)
    # One-sided PSD relative to the SDR centre frequency.
    # ------------------------------------------------------------------
    nperseg = metadata.ms_to_samples(1)     # 1 ms = 18 000 samples per segment
    n_segs  = ANALYSIS_MS                   # 10 non-overlapping segments

    freqs, psd = welch(
        samples,
        fs=SAMPLE_RATE,
        nperseg=nperseg,
        noverlap=0,
        return_onesided=False,              # complex input → two-sided
        detrend=False,
    )

    # Shift so DC is in the centre
    freqs = np.fft.fftshift(freqs)
    psd   = np.fft.fftshift(np.abs(psd))   # take magnitude for complex Welch output
    psd_db = 10 * np.log10(psd + 1e-30)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(freqs / 1e6, psd_db, lw=0.6, color="steelblue")
    ax.set_xlabel("Frequency relative to centre (MHz)")
    ax.set_ylabel("Power spectral density (dB, arbitrary)")
    ax.set_title(
        f"PSD — {ANALYSIS_MS} ms, Welch {nperseg/SAMPLE_RATE*1e3:.0f} ms segments"
        f"  ({n_segs} averages)"
    )
    ax.axvline(RESIDUAL_IF_HZ / 1e6, color="r", lw=0.8, ls="--",
               label=f"Residual IF = {RESIDUAL_IF_HZ/1e3:.0f} kHz")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()

    psd_path = Path("check_capture_psd.png")
    plt.savefig(psd_path, dpi=120)
    plt.close(fig)
    print(f"  PSD saved       : {psd_path.resolve()}")

    _print_section("Done")


if __name__ == "__main__":
    main()