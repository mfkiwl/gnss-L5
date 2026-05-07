"""
debug_acq.py — Synthetic signal diagnostic for GPS L5 acquisition.

Cases:
  1 — Single SV,  residual_IF=0 Hz
  2 — Single SV,  residual_IF=1234 Hz
  3 — Four SVs,   residual_IF=0 Hz     (SNR=35 dB per SV)
  4 — Four SVs,   residual_IF=1234 Hz  (SNR=35 dB per SV)

Run from the project root:
    python scripts/debug_acq.py
"""

import matplotlib
matplotlib.use('Agg')   # WSL: no display
import matplotlib.pyplot as plt
import numpy as np
from dataclasses import dataclass
from typing import List

from gnss_l5.codes.l5_code import L5Code
from gnss_common.acquisition.acq_types import AcqConfig
from gnss_common.acquisition.acquisition import acquire
from gnss_common.capture.capture_metadata import CaptureMetadata
from scipy.signal import fftconvolve
from itertools import combinations

# ---------------------------------------------------------------------------
# Fixed parameters
# ---------------------------------------------------------------------------
SAMPLE_RATE      = 20.46e6
L5_CHIP_RATE_HZ  = 10.23e6
L5_CODE_LENGTH   = 10230
SAMPLES_PER_MS   = round(SAMPLE_RATE / 1000)

CHANNEL          = 'Q5'
COH_INT_MS       = 1
NON_COH_COUNT    = 4
N_SAMPLES        = (NON_COH_COUNT * COH_INT_MS + 1) * SAMPLES_PER_MS

# Single-SV cases
SINGLE_PRN            = 1
SINGLE_CODE_PHASE     = 1000
SINGLE_DOPPLER_HZ     = -2000.0
SINGLE_SNR_DB         = 20.0

# Multi-SV cases — mixed positive and negative Dopplers
MULTI_SNR_DB          = 35.0

@dataclass
class SvSpec:
    prn:               int
    code_phase_samples: int
    doppler_hz:        float

MULTI_SVS = [
    SvSpec(prn=1,  code_phase_samples=1000, doppler_hz=-2000.0),
    SvSpec(prn=7,  code_phase_samples=3500, doppler_hz=+1000.0),
    SvSpec(prn=14, code_phase_samples=6000, doppler_hz=+3000.0),
    SvSpec(prn=21, code_phase_samples=8500, doppler_hz=-1000.0),
]

# Pre-load all codes
_code_cache: dict = {}

def get_chips(prn: int) -> np.ndarray:
    if prn not in _code_cache:
        _code_cache[prn] = L5Code(prn, CHANNEL).chips_bipolar.astype(np.float64)
    return _code_cache[prn]


# ---------------------------------------------------------------------------
# Signal builders
# ---------------------------------------------------------------------------

def make_single_sv_signal(residual_if_hz: float, seed: int = 42) -> np.ndarray:
    """One SV at SINGLE_SNR_DB, complex baseband."""
    rng       = np.random.default_rng(seed)
    t         = np.arange(N_SAMPLES) / SAMPLE_RATE
    amplitude = 10 ** (SINGLE_SNR_DB / 20.0)
    chips     = get_chips(SINGLE_PRN)

    chip_indices = (
        np.floor(
            (np.arange(N_SAMPLES) - SINGLE_CODE_PHASE)
            * L5_CHIP_RATE_HZ / SAMPLE_RATE
        ).astype(int) % L5_CODE_LENGTH
    )
    carrier   = np.exp(1j * 2 * np.pi * (residual_if_hz + SINGLE_DOPPLER_HZ) * t)
    composite = amplitude * chips[chip_indices] * carrier

    noise = (rng.standard_normal(N_SAMPLES) + 1j * rng.standard_normal(N_SAMPLES)) / np.sqrt(2)
    return composite + noise


def make_multi_sv_signal(residual_if_hz: float, seed: int = 42) -> np.ndarray:
    """Four SVs at MULTI_SNR_DB each, complex baseband."""
    rng       = np.random.default_rng(seed)
    t         = np.arange(N_SAMPLES) / SAMPLE_RATE
    amplitude = 10 ** (MULTI_SNR_DB / 20.0)
    composite = np.zeros(N_SAMPLES, dtype=np.complex128)

    for sv in MULTI_SVS:
        chips = get_chips(sv.prn)
        chip_indices = (
            np.floor(
                (np.arange(N_SAMPLES) - sv.code_phase_samples)
                * L5_CHIP_RATE_HZ / SAMPLE_RATE
            ).astype(int) % L5_CODE_LENGTH
        )
        carrier    = np.exp(1j * 2 * np.pi * (residual_if_hz + sv.doppler_hz) * t)
        composite += amplitude * chips[chip_indices] * carrier

    noise = (rng.standard_normal(N_SAMPLES) + 1j * rng.standard_normal(N_SAMPLES)) / np.sqrt(2)
    return composite + noise


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def make_metadata() -> CaptureMetadata:
    return CaptureMetadata(
        sample_rate=SAMPLE_RATE,
        center_freq=0.0,
        dtype=np.complex128,
        is_complex=True,
        num_channels=1,
    )


def make_config(residual_if_hz: float) -> AcqConfig:
    return AcqConfig(
        doppler_search_range_hz=5000.0,
        coh_int_ms=COH_INT_MS,
        non_coh_count=NON_COH_COUNT,
        acq_threshold=1.5,
        residual_if_hz=residual_if_hz,
        return_correlation_surface=True,
    )


def doppler_axis(config: AcqConfig) -> np.ndarray:
    bin_hz = 1000.0 / config.coh_int_ms
    n_bins = int(round(2 * config.doppler_search_range_hz / bin_hz)) + 1
    return np.arange(n_bins) * bin_hz - config.doppler_search_range_hz


def plot_surface(ax_2d, ax_doppler, surface, doppler_offsets,
                 truth_doppler, truth_phase, result, prn):
    im = ax_2d.imshow(
        surface, aspect='auto', origin='lower',
        extent=[0, SAMPLES_PER_MS, doppler_offsets[0], doppler_offsets[-1]],
    )
    ax_2d.axhline(truth_doppler, color='red',    linestyle='--', linewidth=1.0,
                  label=f'Truth D: {truth_doppler:.0f} Hz')
    ax_2d.axvline(truth_phase,   color='orange', linestyle='--', linewidth=1.0,
                  label=f'Truth P: {truth_phase}')
    ax_2d.set_title(f'PRN {prn}  metric={result.peak_metric:.3f}  '
                    f'det={result.detected}')
    ax_2d.set_xlabel('Code phase (samples)')
    ax_2d.set_ylabel('Doppler (Hz)')
    ax_2d.legend(fontsize=7)
    plt.colorbar(im, ax=ax_2d)

    ax_doppler.plot(doppler_offsets, surface[:, result.code_phase_samples],
                    color='steelblue', linewidth=1.0)
    ax_doppler.axvline(truth_doppler,      color='red',    linestyle='--',
                       linewidth=1.0, label=f'Truth: {truth_doppler:.0f} Hz')
    ax_doppler.axvline(result.doppler_hz,  color='orange', linestyle=':',
                       linewidth=1.0, label=f'Det: {result.doppler_hz:.0f} Hz')
    ax_doppler.set_title(f'PRN {prn} — Doppler slice at detected phase')
    ax_doppler.set_xlabel('Doppler (Hz)')
    ax_doppler.set_ylabel('|Correlation|')
    ax_doppler.legend(fontsize=7)


# ---------------------------------------------------------------------------
# Case runners
# ---------------------------------------------------------------------------

def run_single_sv_case(label: str, residual_if_hz: float) -> None:
    print(f"\n{'='*60}")
    print(f"  {label}  (residual_if={residual_if_hz:.0f} Hz, SNR={SINGLE_SNR_DB:.0f} dB)")
    print(f"{'='*60}")

    signal  = make_single_sv_signal(residual_if_hz)
    config  = make_config(residual_if_hz)
    results = acquire(signal, {SINGLE_PRN: get_chips(SINGLE_PRN)}, make_metadata(), config)
    result  = results[SINGLE_PRN]

    print(f"  {'PRN':>4}  {'code_phase':>10}  {'doppler_hz':>10}  "
          f"{'peak_metric':>12}  {'detected':>8}")
    print(f"  {'-'*55}")
    truth_ok = (result.code_phase_samples == SINGLE_CODE_PHASE and
                result.doppler_hz == SINGLE_DOPPLER_HZ)
    flag = '✓' if (result.detected and truth_ok) else '✗'
    print(f"  {SINGLE_PRN:>4}  {result.code_phase_samples:>10}  "
          f"{result.doppler_hz:>10.1f}  {result.peak_metric:>12.4f}  "
          f"{str(result.detected):>8}  {flag}")
    print(f"  {'':>4}  {'(truth='+str(SINGLE_CODE_PHASE)+')':>10}  "
          f"{'('+str(SINGLE_DOPPLER_HZ)+')':>10}")

    d_offsets = doppler_axis(config)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"{label} — SNR={SINGLE_SNR_DB:.0f} dB, "
                 f"residual_IF={residual_if_hz:.0f} Hz  "
                 f"[metric={result.peak_metric:.3f}, detected={result.detected}]",
                 fontsize=10)
    plot_surface(axes[0], axes[1], result.correlation_surface,
                 d_offsets, SINGLE_DOPPLER_HZ, SINGLE_CODE_PHASE, result, SINGLE_PRN)
    fig.tight_layout()
    slug    = label.lower().replace(' ', '_').replace('=','').replace('/','').replace(',','')
    outfile = f'debug_acq_{slug}.png'
    fig.savefig(outfile, dpi=150)
    print(f"  surface plot saved: {outfile}")


def run_multi_sv_case(label: str, residual_if_hz: float) -> None:
    print(f"\n{'='*60}")
    print(f"  {label}  (residual_if={residual_if_hz:.0f} Hz, SNR={MULTI_SNR_DB:.0f} dB/SV)")
    print(f"{'='*60}")

    signal  = make_multi_sv_signal(residual_if_hz)
    config  = make_config(residual_if_hz)
    codes   = {sv.prn: get_chips(sv.prn) for sv in MULTI_SVS}

    # Code cross-correlation sanity check

    chips7  = get_chips(7)
    chips21 = get_chips(21)
    autocorr_21  = np.dot(chips21, chips21)
    autocorr_7   = np.dot(chips7, chips7)
    xcorr        = fftconvolve(chips21, chips7[::-1], mode='full')
    xcorr_max    = np.max(np.abs(xcorr))
    print("\n  Cross-correlation check — all PRN pairs:")
    print(f"  {'PRN pair':>12}  {'max |xcorr|':>12}  {'ratio':>8}")
    print(f"  {'-'*36}")
    prns = [sv.prn for sv in MULTI_SVS]
    for p, q in combinations(prns, 2):
        cp, cq = get_chips(p), get_chips(q)
        xc = fftconvolve(cp, cq[::-1], mode='full')
        ratio = np.max(np.abs(xc)) / L5_CODE_LENGTH
        print(f"  {f'PRN {p} vs PRN {q}':>12}  {np.max(np.abs(xc)):>12.1f}  {ratio:>8.4f}")

    results = acquire(signal, codes, make_metadata(), config)

    r21   = results[21]
    surf  = r21.correlation_surface   # shape (n_doppler_bins, samples_per_coh)
    d_off = doppler_axis(config)
    truth_dop_idx    = int(np.argmin(np.abs(d_off - (-1000.0))))
    detected_dop_idx = int(np.argmin(np.abs(d_off - r21.doppler_hz)))
    print(f"\n  PRN 21 surface diagnostics:")
    print(f"    truth    bin (d=-1000 Hz, phase=8500): {surf[truth_dop_idx,    8500]:.2f}")
    print(f"    detected bin (d={r21.doppler_hz:.0f} Hz, phase={r21.code_phase_samples}): {surf[detected_dop_idx, r21.code_phase_samples]:.2f}")
    print(f"    max in truth    bin: {surf[truth_dop_idx].max():.2f}  at phase {surf[truth_dop_idx].argmax()}")
    print(f"    max in detected bin: {surf[detected_dop_idx].max():.2f}  at phase {surf[detected_dop_idx].argmax()}")

    print(f"  {'PRN':>4}  {'est_phase':>9}  {'tru_phase':>9}  "
          f"{'est_dop':>7}  {'tru_dop':>7}  {'metric':>8}  {'det':>5}")
    print(f"  {'-'*65}")
    all_ok = True
    for sv in MULTI_SVS:
        r = results[sv.prn]
        phase_ok   = abs(r.code_phase_samples - sv.code_phase_samples) <= 1
        doppler_ok = abs(r.doppler_hz - sv.doppler_hz) <= 500
        ok = r.detected and phase_ok and doppler_ok
        flag = '✓' if ok else '✗'
        if not ok:
            all_ok = False
        print(f"  {sv.prn:>4}  {r.code_phase_samples:>9}  "
              f"{sv.code_phase_samples:>9}  {r.doppler_hz:>7.0f}  "
              f"{sv.doppler_hz:>7.0f}  {r.peak_metric:>8.4f}  "
              f"{str(r.detected):>5}  {flag}")

    print(f"\n  Overall: {'ALL PASS ✓' if all_ok else 'FAILURES PRESENT ✗'}")

    # 2x4 figure: top row = 2D surfaces, bottom row = Doppler slices
    d_offsets = doppler_axis(config)
    n_svs     = len(MULTI_SVS)
    fig, axes = plt.subplots(2, n_svs, figsize=(7 * n_svs, 10))
    fig.suptitle(
        f"{label} — {n_svs} SVs, SNR={MULTI_SNR_DB:.0f} dB each, "
        f"residual_IF={residual_if_hz:.0f} Hz",
        fontsize=11,
    )
    for col, sv in enumerate(MULTI_SVS):
        r = results[sv.prn]
        plot_surface(axes[0, col], axes[1, col],
                     r.correlation_surface, d_offsets,
                     sv.doppler_hz, sv.code_phase_samples, r, sv.prn)

    fig.tight_layout()
    slug    = label.lower().replace(' ', '_').replace('=','').replace('/','').replace(',','')
    outfile = f'debug_acq_{slug}.png'
    fig.savefig(outfile, dpi=150)
    print(f"  surface plot saved: {outfile}")


# ---------------------------------------------------------------------------
# Run all cases
# ---------------------------------------------------------------------------
run_single_sv_case('Case 1 — residual_IF=0 Hz',    residual_if_hz=0.0)
run_single_sv_case('Case 2 — residual_IF=1234 Hz', residual_if_hz=1234.0)
run_multi_sv_case( 'Case 3 — residual_IF=0 Hz',    residual_if_hz=0.0)
run_multi_sv_case( 'Case 4 — residual_IF=1234 Hz', residual_if_hz=1234.0)