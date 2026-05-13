"""
FFT-based parallel code-phase acquisition function.

Algorithm overview
------------------
For each Doppler hypothesis, the full circular cross-correlation across all
code phase offsets is computed simultaneously via a single IFFT (parallel
code-phase search).  Doppler hypotheses are evaluated by shifting the signal
FFT in the frequency domain rather than by explicit carrier wipeoff, which
avoids recomputing the signal FFT for each bin.

The residual IF (SDR front-end tuning offset from nominal signal frequency)
is wiped off once on the full signal before any further processing.  The time
vector spans the full signal so that phase is continuous across segment
boundaries.  The Doppler step size is derived from the coherent integration
time and is not a free parameter.

GPU acceleration
----------------
All array operations are dispatched through the ``xp`` namespace, which
defaults to numpy.  Pass ``xp=cupy`` to execute on a CUDA GPU.  The caller
is responsible for moving data to device memory before calling this function.
"""

from __future__ import annotations

import numpy as np

from gnss_common.acquisition.acq_types import AcqConfig, AcqResult
from gnss_common.capture.capture_metadata import CaptureMetadata


def acquire(
    signal: np.ndarray,
    codes: dict[int, np.ndarray],
    metadata: CaptureMetadata,
    config: AcqConfig,
    xp=np,
) -> dict[int, AcqResult]:
    """
    Perform coarse FFT-based acquisition for one or more PRNs.

    Parameters
    ----------
    signal : np.ndarray
        1-D complex array of baseband I/Q samples (dtype complex64 or
        complex128).  Typically read from a capture file as int8 I/Q pairs
        and converted to complex by the caller.  Must contain at least
        non_coh_count * coh_int_ms milliseconds of data.
    codes : dict[int, np.ndarray]
        Mapping of PRN integer to bipolar chip array (values ±1).
        Each array must have length equal to one code period (10230 chips
        for GPS L5).
    metadata : CaptureMetadata
        Signal capture parameters.  Uses sample_rate and center_freq.
    config : AcqConfig
        Acquisition configuration.  See AcqConfig for field descriptions.
    xp : module, optional
        Array namespace.  Defaults to numpy.  Pass cupy for GPU execution.

    Returns
    -------
    dict[int, AcqResult]
        Mapping of PRN integer to AcqResult.  Keys match the input codes dict.
    """
    if not codes:
        return {}

    #------------------------------------------------------------------
    # Check that all code arrays have the same length, which is needed for
    # building the code replica matrix.  (This is not a fundamental requirement
    # of the algorithm, but it simplifies the implementation)
    #------------------------------------------------------------------
    prn_list = list(codes.keys())
    code_len_chips = len(codes[prn_list[0]])
    if not all(len(c) == code_len_chips for c in codes.values()):
        raise ValueError("all codes must have the same length")
    
    # ------------------------------------------------------------------
    # Derived parameters
    # ------------------------------------------------------------------
    sample_rate      = metadata.sample_rate
    samples_per_ms   = round(sample_rate / 1000)
    # Samples per coherent integration segment (one code period at coh_int_ms)
    samples_per_coh  = samples_per_ms * config.coh_int_ms
    # Total samples consumed by the acquisition
    samples_required = samples_per_coh * config.non_coh_count

    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------
    if len(signal) < samples_required:
        raise ValueError(
            f"signal length {len(signal)} is insufficient: need at least "
            f"{samples_required} samples for non_coh_count="
            f"{config.non_coh_count} x coh_int_ms={config.coh_int_ms} ms "
            f"at sample_rate={sample_rate} Hz"
        )

    # ------------------------------------------------------------------
    # Residual IF wipeoff — applied once to the full signal.
    # Time vector spans all samples_required samples so that phase is
    # continuous across coherent segment boundaries.
    # ------------------------------------------------------------------
    t_full    = xp.arange(samples_required) / sample_rate
    if_phasor = xp.exp(-1j * 2 * np.pi * config.residual_if_hz * t_full).astype(xp.complex64)
    signal_bb = xp.asarray(signal[:samples_required], dtype=xp.complex64) * if_phasor

    # ------------------------------------------------------------------
    # Doppler search grid
    # Bin spacing = 1 / T_coh = 1000 / coh_int_ms Hz — derived, not configured
    # ------------------------------------------------------------------
    doppler_bin_hz  = 1000.0 / config.coh_int_ms
    n_doppler_bins  = int(round(
        2 * config.doppler_search_range_hz / doppler_bin_hz
    )) + 1
    # Doppler hypotheses centred on zero, symmetric about the search range
    doppler_offsets = (
        np.arange(n_doppler_bins) * doppler_bin_hz
        - config.doppler_search_range_hz
    )  # shape (n_doppler_bins,), Hz
    fft_length = 2 * samples_per_coh

    # ------------------------------------------------------------------
    # Build code replica matrix and compute conjugate FFTs.
    # Each row is one PRN's chip sequence sampled at sample_rate,
    # zero-padded to 2 * samples_per_coh to prevent circular wrap-around.
    # Shape before padding : (n_prns, samples_per_coh)
    # Shape after padding  : (n_prns, fft_length)
    # ------------------------------------------------------------------

    n_prns     = len(prn_list)
    
    # Map sample indices to chip indices for one coherent segment
    chip_indices = (
        np.arange(samples_per_coh)
        * (code_len_chips * config.coh_int_ms / samples_per_coh)
    ).astype(int) % code_len_chips  # shape (samples_per_coh,)

    # Code replica matrix, zero-padded: shape (n_prns, fft_length)
    code_matrix = xp.zeros((n_prns, fft_length), dtype=xp.float32)
    # TODO: numpy silently promotes fft.fft to complex128
    # we should probably use scipy fftpack to preserve complex64 precision 
    # Primary concern is execution speed.
    for i, prn in enumerate(prn_list):
        chips = xp.asarray(codes[prn], dtype=xp.float32)
        code_matrix[i, :samples_per_coh] = chips[chip_indices]

    # Conjugate FFT of each code replica: shape (n_prns, fft_length)
    code_fft_conj = xp.conj(xp.fft.fft(code_matrix, axis=1))

    # ------------------------------------------------------------------
    # Non-coherent accumulation of correlation surface.
    # Shape: (n_prns, n_doppler_bins, samples_per_coh)
    # Only the first samples_per_coh columns of each IFFT are retained;
    # the second half contains wrap-around artefacts from zero-padding.
    # ------------------------------------------------------------------
    surface = xp.zeros(
        (n_prns, n_doppler_bins, samples_per_coh), dtype=xp.float32
    )

    for seg_idx in range(config.non_coh_count):
        seg_start = seg_idx * samples_per_coh
        seg_end   = seg_start + samples_per_coh

        # Extract pre-whitened segment and zero-pad for FFT
        segment_padded = xp.zeros(fft_length, dtype=xp.complex64)
        segment_padded[:samples_per_coh] = signal_bb[seg_start:seg_end]
        signal_fft = xp.fft.fft(segment_padded)  # shape (fft_length,)
        signal_fft[0] = 0.0  # Remove potential DC spike to reduce noise floor and false peaks

        # Correlate against all code replicas for each Doppler hypothesis
        for dop_idx, dop_hz in enumerate(doppler_offsets):
            # Shift signal FFT by bin_shift bins — equivalent to multiplying
            # by exp(-j*2*pi*dop_hz*t) in the time domain
            bin_shift          = int(round(dop_hz / (sample_rate / fft_length)))
            signal_fft_shifted = xp.roll(signal_fft, -bin_shift)

            # Multiply by conjugate code FFTs and IFFT all PRNs at once.
            # Result shape: (n_prns, fft_length)
            corr = xp.fft.ifft(
                code_fft_conj * signal_fft_shifted[xp.newaxis, :],
                axis=1,
            )

            # Non-coherent accumulation: magnitude, valid half only
            surface[:, dop_idx, :] += xp.abs(corr[:, :samples_per_coh])

    # ------------------------------------------------------------------
    # Peak detection — one AcqResult per PRN
    # ------------------------------------------------------------------
    results = {}
    exclusion_zone = 3   # Exclude ±3 chips around the primary peak when searching for the second peak
    samples_per_chip = samples_per_ms / code_len_chips
    exclusion_half_width = max(1, round(exclusion_zone * samples_per_chip))

    for i, prn in enumerate(prn_list):
        prn_surface = surface[i]  # shape (n_doppler_bins, samples_per_coh)

        # Global peak location
        peak_dop_idx   = int(xp.argmax(xp.max(prn_surface, axis=1)))
        peak_phase_idx = int(xp.argmax(prn_surface[peak_dop_idx]))
        peak_value     = float(prn_surface[peak_dop_idx, peak_phase_idx])

        # Second-peak ratio: mask a neighbourhood around the primary peak
        # across all Doppler bins before searching for the next highest value

        masked = prn_surface.copy()
        lo = max(0, peak_phase_idx - exclusion_half_width)
        hi = min(samples_per_coh, peak_phase_idx + exclusion_half_width + 1)
        masked[:, lo:hi] = 0.0
        # Handle wrap-around exclusion if the primary peak is near the edges of the code period
        if peak_phase_idx - exclusion_half_width < 0:
            masked[:, samples_per_coh + (peak_phase_idx - exclusion_half_width):] = 0.0
        if peak_phase_idx + exclusion_half_width + 1 > samples_per_coh:
            masked[:, :(peak_phase_idx + exclusion_half_width + 1 - samples_per_coh)] = 0.0

        second_peak_value = float(xp.max(masked))

        # Guard against divide-by-zero in pure-noise edge case
        peak_metric = (
            peak_value / second_peak_value
            if second_peak_value > 0
            else float('inf')
        )

        detected = peak_metric > config.acq_threshold

        # ------------------------------------------------------------------
        # Carrier phase at the peak.
        # Extracted from the complex IFFT output of the last non-coherent
        # segment at the detected Doppler bin.  Phase is not meaningful to
        # average non-coherently, so we use a single segment.
        # ------------------------------------------------------------------
        last_start = (config.non_coh_count - 1) * samples_per_coh
        last_end   = last_start + samples_per_coh

        segment_padded = xp.zeros(fft_length, dtype=xp.complex64)
        segment_padded[:samples_per_coh] = signal_bb[last_start:last_end]
        signal_fft     = xp.fft.fft(segment_padded)
        signal_fft[0] = 0.0  # Remove potential DC spike to reduce noise floor and false peaks

        dop_hz        = float(doppler_offsets[peak_dop_idx])
        bin_shift     = int(round(dop_hz / (sample_rate / fft_length)))
        corr_complex  = xp.fft.ifft(
            code_fft_conj[i] * xp.roll(signal_fft, -bin_shift)
        )
        doppler_phase_rad = float(xp.angle(corr_complex[peak_phase_idx]))

        results[prn] = AcqResult(
            code_phase_samples=peak_phase_idx,
            doppler_hz=dop_hz,
            doppler_phase_rad=doppler_phase_rad,
            peak_metric=peak_metric,
            detected=detected,
            correlation_surface=(
                np.array(prn_surface)
                if config.return_correlation_surface
                else None
            ),
        )

    return results