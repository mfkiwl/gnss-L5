"""
Dataclasses for GPS L5 acquisition configuration and results.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class AcqConfig:
    """
    Configuration parameters for the acquisition function.

    Attributes
    ----------
    doppler_search_range_hz : float
        Half-width of the Doppler search band in Hz.
        The search covers [-doppler_search_range_hz, +doppler_search_range_hz].
    coh_int_ms : int
        Coherent integration time in milliseconds. Must be a positive integer.
        Doppler bin spacing is derived as 1000 / coh_int_ms Hz.
    non_coh_count : int
        Number of non-coherent integrations. Must be positive.
    acq_threshold : float
        Peak-to-second-peak ratio threshold for detection. Must be positive.
    residual_if_hz : float
        Residual intermediate frequency in Hz — the difference between the
        signal's nominal carrier frequency and the SDR front-end centre
        frequency.  Computed by the caller as ``nominal_freq_hz - center_freq``.
        Wiped off once before the Doppler search loop.  Defaults to 0.0 (no
        residual IF).
    return_correlation_surface : bool
        If True, the full 2D correlation surface (Doppler bins x code phase
        samples) is retained in AcqResult. If False (default), it is computed
        internally but discarded to save memory.
    """

    doppler_search_range_hz: float
    coh_int_ms: int
    non_coh_count: int
    acq_threshold: float
    residual_if_hz: float = 0.0
    return_correlation_surface: bool = False

    def __post_init__(self):
        if not isinstance(self.coh_int_ms, int):
            raise TypeError(
                f"coh_int_ms must be an integer, got {type(self.coh_int_ms).__name__}"
            )
        if self.coh_int_ms <= 0:
            raise ValueError(
                f"coh_int_ms must be a positive integer, got {self.coh_int_ms}"
            )
        if self.non_coh_count <= 0:
            raise ValueError(
                f"non_coh_count must be positive, got {self.non_coh_count}"
            )
        if self.doppler_search_range_hz <= 0:
            raise ValueError(
                f"doppler_search_range_hz must be positive, "
                f"got {self.doppler_search_range_hz}"
            )
        if self.acq_threshold <= 0:
            raise ValueError(
                f"acq_threshold must be positive, got {self.acq_threshold}"
            )


@dataclass
class AcqResult:
    """
    Result of acquisition for a single PRN.

    Attributes
    ----------
    code_phase_samples : int
        Sample index of the correlation peak within one code period.
    doppler_hz : float
        Doppler frequency offset relative to nominal L5 center frequency, in Hz.
    doppler_phase_rad : float
        Carrier phase at the correlation peak, in radians. Range: (-pi, pi].
        Provides the tracking loop with an initial phase estimate.
    peak_metric : float
        Peak-to-second-peak ratio of the correlation surface. Used as the
        detection statistic and a coarse SNR proxy.
    detected : bool
        True if peak_metric exceeds the configured acquisition threshold.
    correlation_surface : np.ndarray or None
        2D array of shape (n_doppler_bins, samples_per_ms) containing the
        non-coherently accumulated correlation magnitude across all Doppler
        and code phase hypotheses. None if return_correlation_surface was
        False in AcqConfig.
    """

    code_phase_samples: int
    doppler_hz: float
    doppler_phase_rad: float
    peak_metric: float
    detected: bool
    correlation_surface: Optional[np.ndarray] = field(default=None, repr=False)