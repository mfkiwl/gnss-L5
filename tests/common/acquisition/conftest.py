import numpy as np
import pytest
from dataclasses import dataclass
from gnss_l5.codes.l5_code import L5Code
from gnss_common.capture.capture_metadata import CaptureMetadata

# L5_NOMINAL_FREQ will eventually live in gnss_l5/ as a module constant
L5_NOMINAL_FREQ_HZ = 1176.45e6
L5_CHIP_RATE_HZ    = 10.23e6
L5_CODE_LENGTH     = 10230

# Fixture parameters
_SAMPLE_RATE  = 20.46e6
_CENTER_FREQ  = L5_NOMINAL_FREQ_HZ - 1234.0  # residual IF of +1234 Hz
_RESIDUAL_IF  = L5_NOMINAL_FREQ_HZ - _CENTER_FREQ
_SNR_DB       = 20.0
_COH_INT_MS   = 1
_NON_COH_COUNT = 4


@dataclass(frozen=True)
class _SvTruth:
    """Ground truth parameters for one SV in the synthetic signal."""
    prn: int
    code_phase_samples: int
    doppler_hz: float  # exact multiple of bin spacing (1000 Hz) for clean assertions


# Four SVs: well-separated code phases, Dopplers at exact bin centres
_SV_TRUTH = [
    _SvTruth(prn=1,  code_phase_samples=1000, doppler_hz=-2000.0),
    _SvTruth(prn=7,  code_phase_samples=3500, doppler_hz= 1000.0),
    _SvTruth(prn=14, code_phase_samples=6000, doppler_hz= 3000.0),
    _SvTruth(prn=21, code_phase_samples=8500, doppler_hz=-1000.0),
]


def _make_signal(sv_truth, sample_rate, residual_if, snr_db, n_samples, rng):
    """
    Generate a composite real IF signal containing one L5 Q5 SV.
    Signal = A * code(t - tau) * exp(j*2π*(IF + doppler)*t) + AWGN
    """
    t = np.arange(n_samples) / sample_rate
    amplitude = 10 ** (snr_db / 20.0)

    # Chip index for each sample, accounting for code phase delay
    chip_indices = (
        np.floor(
            (np.arange(n_samples) - sv_truth.code_phase_samples)
            * L5_CHIP_RATE_HZ / sample_rate
        ).astype(int) % L5_CODE_LENGTH
    )

    chips = L5Code(sv_truth.prn, 'Q5').chips_bipolar.astype(np.float64)
    sampled_code = chips[chip_indices]

    carrier = np.exp(1j * 2 * np.pi * (residual_if + sv_truth.doppler_hz) * t)

    return amplitude * sampled_code * carrier


@pytest.fixture
def acq_fixture():
    """
    Composite complex I/Q signal containing four L5 Q5 SVs plus AWGN.

    Returns:
        signal   : np.ndarray, complex128, shape (n_samples,)
        codes    : dict[int, np.ndarray]  PRN -> bipolar chip array (10230,)
        metadata : CaptureMetadata
        sv_truth : list[_SvTruth]  ground truth for assertions
    """
    rng = np.random.default_rng(42)

    samples_per_ms = round(_SAMPLE_RATE / 1000)
    # Extra ms of headroom so the function isn't working right at the edge
    n_samples = (_NON_COH_COUNT * _COH_INT_MS + 1) * samples_per_ms

    composite = np.zeros(n_samples, dtype=np.complex128)
    for sv in _SV_TRUTH:
        composite += _make_signal(
            sv, _SAMPLE_RATE, _RESIDUAL_IF, _SNR_DB, n_samples, rng
        )

    # Unit-variance complex AWGN (independent I and Q)
    composite += (rng.standard_normal(n_samples) + 1j * rng.standard_normal(n_samples)) / np.sqrt(2)

    codes = {
        sv.prn: L5Code(sv.prn, 'Q5').chips_bipolar.astype(np.float64)
        for sv in _SV_TRUTH
    }

    metadata = CaptureMetadata(
        sample_rate=_SAMPLE_RATE,
        center_freq=_CENTER_FREQ,
        dtype=np.complex128,
        is_complex=True,
        num_channels=1,
    )

    return composite, codes, metadata, _SV_TRUTH