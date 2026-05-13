"""
run_acquisition.py — GPS L5 acquisition against a real capture file.

Reads the minimum required data (coh_int_ms × non_coh_count milliseconds),
generates Q5 pilot codes for all known L5-capable PRNs, runs the parallel
FFT acquisition, and reports results.  For each detected SV, a two-panel
plot of the correlation surface is saved to disk.

Run from the project root:
    python scripts/run_acquisition.py

Output images are written to the current working directory.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless — no display required (WSL-safe)
import matplotlib.pyplot as plt

from gnss_common.capture.capture_file import CaptureFile
from gnss_common.capture.capture_metadata import CaptureMetadata
from gnss_common.acquisition.acq_types import AcqConfig
from gnss_common.acquisition.acquisition import acquire
from gnss_l5.codes.l5_code import L5Code
from gnss_l5.constants import (
    L5_NOMINAL_FREQ_HZ,
    L5_CHIP_RATE_HZ,
    L5_CAPABLE_PRNS,
)

# ---------------------------------------------------------------------------
# Capture-specific constants — edit to match the file under test
# ---------------------------------------------------------------------------

CAPTURE_PATH:    Path  = Path("/mnt/e/L5_capture/L5b_IF20KHz_FS18MHz/L5b_IF20Khz_FS18MHz.bin")
#CAPTURE_PATH:    Path  = Path("/mnt/e/L5_capture/test_L5_1.bin")
SAMPLE_RATE:     float = 18.0e6   # Hz
RESIDUAL_IF_HZ:  float = 20.0e3  # Hz — hardware tuning offset
CENTER_FREQ:     float = L5_NOMINAL_FREQ_HZ + RESIDUAL_IF_HZ

# ---------------------------------------------------------------------------
# Acquisition configuration
# ---------------------------------------------------------------------------

COH_INT_MS:            int   = 1        # ms — 1 code period
NON_COH_COUNT:         int   = 1       # accumulations — 10 ms total
DOPPLER_SEARCH_HZ:     float = 10000.0  # half-width, Hz  (±5 kHz)
ACQ_THRESHOLD:         float = 2.5     # peak-to-second-peak ratio for detection


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _print_section(title: str) -> None:
    print(f"\n{'═' * 70}")
    print(f"  {title}")
    print(f"{'═' * 70}")


def _build_codes() -> dict[int, np.ndarray]:
    """Return Q5 bipolar chip arrays for every L5-capable PRN."""
    print(f"  Generating Q5 codes for {len(L5_CAPABLE_PRNS)} PRNs …", end=" ", flush=True)
    codes = {
        prn: L5Code(prn, "Q5").chips_bipolar.astype(np.float64)
        for prn in L5_CAPABLE_PRNS
    }
    print("done")
    return codes


def _print_results(results: dict[int, object], sample_rate: float) -> None:
    detected = [prn for prn, r in results.items() if r.detected]

    print(f"\n  {len(detected)} of {len(results)} PRNs detected\n")
    print(
        f"  {'PRN':>4}  {'Det':>4}  {'Doppler (Hz)':>13}  "
        f"{'Phase (samp)':>13}  {'Phase (chips)':>14}  {'Peak metric':>12}"
    )
    print(f"  {'─'*4}  {'─'*4}  {'─'*13}  {'─'*13}  {'─'*14}  {'─'*12}")

    # Detected first (sorted by peak metric), then non-detected
    rows_detected     = sorted(
        ((prn, r) for prn, r in results.items() if r.detected),
        key=lambda t: t[1].peak_metric,
        reverse=True,
    )
    rows_not_detected = sorted(
        ((prn, r) for prn, r in results.items() if not r.detected),
        key=lambda t: t[0],
    )

    for prn, r in rows_detected + rows_not_detected:
        code_phase_chips = r.code_phase_samples * L5_CHIP_RATE_HZ / sample_rate
        print(
            f"  {prn:>4}  {'YES':>4}  {r.doppler_hz:>+13.1f}  "
            f"{r.code_phase_samples:>13}  {code_phase_chips:>14.1f}  "
            f"{r.peak_metric:>12.3f}"
            if r.detected else
            f"  {prn:>4}  {'no':>4}  {r.doppler_hz:>+13.1f}  "
            f"{r.code_phase_samples:>13}  {code_phase_chips:>14.1f}  "
            f"{r.peak_metric:>12.3f}"
        )


def _plot_surface(
    prn: int,
    result,
    config: AcqConfig,
    sample_rate: float,
) -> None:
    """Save a two-panel correlation surface plot for one detected SV."""
    surface = result.correlation_surface       # (n_doppler_bins, samples_per_coh)
    n_doppler_bins, samples_per_coh = surface.shape

    doppler_bin_hz = 1000.0 / config.coh_int_ms
    doppler_axis   = (
        np.arange(n_doppler_bins) * doppler_bin_hz - config.doppler_search_range_hz
    )                                          # Hz
    code_phase_axis_chips = (
        np.arange(samples_per_coh) * L5_CHIP_RATE_HZ / sample_rate
    )                                          # chips

    fig, (ax_map, ax_slice) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f"PRN {prn:02d}   Doppler {result.doppler_hz:+.0f} Hz   "
        f"Code phase {result.code_phase_samples} samp "
        f"({result.code_phase_samples * L5_CHIP_RATE_HZ / sample_rate:.1f} chips)   "
        f"Peak metric {result.peak_metric:.3f}",
        fontsize=10,
    )

    # --- 2-D heat map ---
    im = ax_map.imshow(
        surface,
        aspect="auto",
        origin="lower",
        extent=[
            code_phase_axis_chips[0], code_phase_axis_chips[-1],
            doppler_axis[0] / 1e3,    doppler_axis[-1] / 1e3,
        ],
        interpolation="nearest",
    )
    ax_map.set_xlabel("Code phase (chips)")
    ax_map.set_ylabel("Doppler (kHz)")
    ax_map.set_title("Non-coherent correlation surface")
    fig.colorbar(im, ax=ax_map, label="Magnitude")
    ax_map.axhline(
        result.doppler_hz / 1e3, color="r", lw=0.8, ls="--", label="peak Doppler"
    )
    peak_chips = result.code_phase_samples * L5_CHIP_RATE_HZ / sample_rate
    ax_map.axvline(peak_chips, color="orange", lw=0.8, ls="--", label="peak code phase")
    ax_map.legend(fontsize=7, loc="upper right")

    # --- Code-phase slice at peak Doppler ---
    peak_dop_idx = int(round(
        (result.doppler_hz + config.doppler_search_range_hz) / doppler_bin_hz
    ))
    # Clamp to valid range in case of rounding
    peak_dop_idx = max(0, min(peak_dop_idx, n_doppler_bins - 1))

    ax_slice.plot(code_phase_axis_chips, surface[peak_dop_idx], lw=0.7, color="steelblue")
    ax_slice.axvline(
        peak_chips, color="r", lw=0.8, ls="--",
        label=f"peak = {result.code_phase_samples} samp ({peak_chips:.1f} chips)",
    )
    ax_slice.set_xlabel("Code phase (chips)")
    ax_slice.set_ylabel("Correlation magnitude")
    ax_slice.set_title(f"Code-phase slice at Doppler = {result.doppler_hz:+.0f} Hz")
    ax_slice.legend(fontsize=7)
    ax_slice.grid(alpha=0.3)

    plt.tight_layout()
    out_path = Path(f"acq_surface_prn{prn:02d}.png")
    plt.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"    Surface plot : {out_path.resolve()}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:

    # ------------------------------------------------------------------
    # Open capture file
    # ------------------------------------------------------------------
    _print_section("Capture file")
    if not CAPTURE_PATH.exists():
        print(f"  ERROR: file not found: {CAPTURE_PATH}", file=sys.stderr)
        sys.exit(1)

    cf = CaptureFile(CAPTURE_PATH, dtype=np.int8, is_complex=True)
    metadata = CaptureMetadata(
        sample_rate=SAMPLE_RATE,
        center_freq=CENTER_FREQ,
        dtype=np.int8,
        is_complex=True,
    )

    n_ms      = COH_INT_MS * NON_COH_COUNT
    n_samples = metadata.ms_to_samples(n_ms)

    print(f"  Path          : {CAPTURE_PATH}")
    print(f"  Total samples : {cf.num_samples:,}  ({cf.num_samples / SAMPLE_RATE:.1f} s)")
    print(f"  Reading       : {n_ms} ms = {n_samples:,} samples")

    t0 = time.perf_counter()
    signal = cf.read(count=n_samples)
    print(f"  Read time     : {time.perf_counter() - t0:.2f} s")

    # ------------------------------------------------------------------
    # Code generation
    # ------------------------------------------------------------------
    _print_section("Code generation")
    codes = _build_codes()

    # ------------------------------------------------------------------
    # Acquisition
    # ------------------------------------------------------------------
    _print_section("Acquisition")
    doppler_bin_hz = 1000.0 / COH_INT_MS
    n_doppler_bins = int(round(2 * DOPPLER_SEARCH_HZ / doppler_bin_hz)) + 1

    print(f"  Coherent integration : {COH_INT_MS} ms")
    print(f"  Non-coh. accum.      : {NON_COH_COUNT}  ({n_ms} ms total)")
    print(f"  Doppler search       : ±{DOPPLER_SEARCH_HZ:.0f} Hz  "
          f"({n_doppler_bins} bins × {doppler_bin_hz:.0f} Hz)")
    print(f"  Residual IF          : {RESIDUAL_IF_HZ / 1e3:.1f} kHz")
    print(f"  Detection threshold  : {ACQ_THRESHOLD}")
    print(f"  PRNs searched        : {len(codes)}")

    config = AcqConfig(
        doppler_search_range_hz=DOPPLER_SEARCH_HZ,
        coh_int_ms=COH_INT_MS,
        non_coh_count=NON_COH_COUNT,
        acq_threshold=ACQ_THRESHOLD,
        residual_if_hz=RESIDUAL_IF_HZ,
        return_correlation_surface=True,   # needed for surface plots
    )

    print("\n  Running …", end=" ", flush=True)
    t0 = time.perf_counter()
    results = acquire(signal, codes, metadata, config)
    elapsed = time.perf_counter() - t0
    print(f"done  ({elapsed:.1f} s)")

    # ------------------------------------------------------------------
    # Results table
    # ------------------------------------------------------------------
    _print_section("Results")
    _print_results(results, SAMPLE_RATE)

    # ------------------------------------------------------------------
    # Correlation surface plots for detected SVs
    # ------------------------------------------------------------------
    detected = {prn: r for prn, r in results.items() if r.detected}

    if detected:
        _print_section(f"Surface plots  ({len(detected)} SVs)")
        for prn in sorted(detected.keys()):
            _plot_surface(prn, detected[prn], config, SAMPLE_RATE)
    else:
        print("\n  No SVs detected — adjust threshold or check capture file.")

    _print_section("Done")


if __name__ == "__main__":
    main()