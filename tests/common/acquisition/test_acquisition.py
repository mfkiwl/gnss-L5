import numpy as np
import pytest
from gnss_common.acquisition.acq_types import AcqConfig, AcqResult
from gnss_common.acquisition.acquisition import acquire

# Tolerance for code phase assertion: within 1 sample of truth
_CODE_PHASE_TOL_SAMPLES = 1

# Doppler bin spacing at 1ms coherent integration
_DOPPLER_BIN_HZ = 1000.0

# Must match conftest._RESIDUAL_IF — both derived from L5_NOMINAL - center_freq
_RESIDUAL_IF_HZ = 1234.0

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(**kwargs):
    defaults = dict(
        doppler_search_range_hz=5000.0,
        coh_int_ms=1,
        non_coh_count=4,
        acq_threshold=1.5,
        return_correlation_surface=False,
    )
    defaults.update(kwargs)
    return AcqConfig(**defaults)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

class TestAcquireInputValidation:

    def test_signal_too_short_raises(self, acq_fixture):
        signal, codes, metadata, sv_truth = acq_fixture
        config = _make_config(coh_int_ms=1, non_coh_count=4)
        # One sample short of required length
        samples_per_ms = round(metadata.sample_rate / 1000)
        required = config.non_coh_count * config.coh_int_ms * samples_per_ms
        short_signal = signal[:required - 1]
        with pytest.raises(ValueError):
            acquire(short_signal, codes, metadata, config)

    def test_signal_longer_than_required_is_accepted(self, acq_fixture):
        signal, codes, metadata, sv_truth = acq_fixture
        config = _make_config()
        # Should not raise
        results = acquire(signal, codes, metadata, config)
        assert isinstance(results, dict)

    def test_empty_codes_dict_returns_empty_results(self, acq_fixture):
        signal, codes, metadata, sv_truth = acq_fixture
        config = _make_config()
        results = acquire(signal, {}, metadata, config)
        assert results == {}


# ---------------------------------------------------------------------------
# Noise-only: no signal injected for searched PRNs
# ---------------------------------------------------------------------------

class TestAcquireNoSignal:

    def test_noise_only_nothing_detected(self, acq_fixture):
        """
        Search for PRNs not present in the composite signal.
        All should be undetected.
        """
        signal, codes, metadata, sv_truth = acq_fixture
        config = _make_config()

        # PRNs 30, 31, 32 are not in the composite signal
        from gnss_l5.codes.l5_code import L5Code
        absent_codes = {
            prn: L5Code(prn, 'Q5').chips_bipolar.astype(np.float64)
            for prn in [30, 31, 32]
        }
        results = acquire(signal, absent_codes, metadata, config)

        for prn, result in results.items():
            assert not result.detected, f"PRN {prn} falsely detected"


# ---------------------------------------------------------------------------
# Signal present: detection and parameter accuracy
# ---------------------------------------------------------------------------

class TestAcquireDetection:

    def test_all_svs_detected(self, acq_fixture):
        signal, codes, metadata, sv_truth = acq_fixture
        config = _make_config()
        results = acquire(signal, codes, metadata, config)

        for sv in sv_truth:
            assert sv.prn in results
            assert results[sv.prn].detected, f"PRN {sv.prn} not detected"

    def test_code_phase_accuracy(self, acq_fixture):
        signal, codes, metadata, sv_truth = acq_fixture
        config = _make_config()
        results = acquire(signal, codes, metadata, config)

        for sv in sv_truth:
            estimated = results[sv.prn].code_phase_samples
            assert abs(estimated - sv.code_phase_samples) <= _CODE_PHASE_TOL_SAMPLES, (
                f"PRN {sv.prn}: code phase {estimated} not within "
                f"{_CODE_PHASE_TOL_SAMPLES} sample(s) of truth {sv.code_phase_samples}"
            )

    def test_doppler_accuracy(self, acq_fixture):
        """
        Injected Dopplers are exact multiples of bin spacing,
        so estimated Doppler should match exactly.
        """
        signal, codes, metadata, sv_truth = acq_fixture
        config = _make_config(residual_if_hz=_RESIDUAL_IF_HZ)
        results = acquire(signal, codes, metadata, config)

        for sv in sv_truth:
            estimated = results[sv.prn].doppler_hz
            assert estimated == pytest.approx(sv.doppler_hz, abs=_DOPPLER_BIN_HZ / 2), (
                f"PRN {sv.prn}: doppler {estimated} Hz not within "
                f"half a bin of truth {sv.doppler_hz} Hz"
            )

    def test_peak_metric_above_threshold(self, acq_fixture):
        signal, codes, metadata, sv_truth = acq_fixture
        config = _make_config()
        results = acquire(signal, codes, metadata, config)

        for sv in sv_truth:
            assert results[sv.prn].peak_metric > config.acq_threshold, (
                f"PRN {sv.prn}: peak metric {results[sv.prn].peak_metric} "
                f"not above threshold {config.acq_threshold}"
            )

    def test_doppler_phase_is_float_in_range(self, acq_fixture):
        """Carrier phase at peak should be in (-pi, pi]."""
        signal, codes, metadata, sv_truth = acq_fixture
        config = _make_config()
        results = acquire(signal, codes, metadata, config)

        for sv in sv_truth:
            phase = results[sv.prn].doppler_phase_rad
            assert -np.pi < phase <= np.pi, (
                f"PRN {sv.prn}: phase {phase} outside (-pi, pi]"
            )

    def test_result_keys_match_input_codes(self, acq_fixture):
        signal, codes, metadata, sv_truth = acq_fixture
        config = _make_config()
        results = acquire(signal, codes, metadata, config)
        assert set(results.keys()) == set(codes.keys())


# ---------------------------------------------------------------------------
# Correlation surface
# ---------------------------------------------------------------------------

class TestCorrelationSurface:

    def test_correlation_surface_none_by_default(self, acq_fixture):
        signal, codes, metadata, sv_truth = acq_fixture
        config = _make_config(return_correlation_surface=False)
        results = acquire(signal, codes, metadata, config)

        for result in results.values():
            assert result.correlation_surface is None

    def test_correlation_surface_returned_when_requested(self, acq_fixture):
        signal, codes, metadata, sv_truth = acq_fixture
        config = _make_config(return_correlation_surface=True)
        results = acquire(signal, codes, metadata, config)

        for result in results.values():
            assert result.correlation_surface is not None

    def test_correlation_surface_shape(self, acq_fixture):
        """
        Surface shape should be (n_doppler_bins, samples_per_ms).
        n_doppler_bins = 2 * doppler_search_range_hz / bin_spacing + 1
        """
        signal, codes, metadata, sv_truth = acq_fixture
        config = _make_config(
            doppler_search_range_hz=5000.0,
            coh_int_ms=1,
            return_correlation_surface=True,
        )
        results = acquire(signal, codes, metadata, config)

        samples_per_ms = round(metadata.sample_rate / 1000)
        bin_spacing = 1000.0 / config.coh_int_ms
        expected_doppler_bins = int(
            round(2 * config.doppler_search_range_hz / bin_spacing)
        ) + 1

        for result in results.values():
            assert result.correlation_surface.shape == (
                expected_doppler_bins, samples_per_ms
            )


# ---------------------------------------------------------------------------
# Complex IQ input
# ---------------------------------------------------------------------------

class TestComplexIQInput:
    """
    Guards the complex IQ signal path through acquire().

    The regression being protected against: the dtype=float64 cast on
    signal_bb that was removed from acquire().  That cast silently discards
    the imaginary part of any complex input, losing half the signal energy.

    Strategy: rotate the fixture's real signal by 90 degrees (multiply by j)
    so all energy moves to the imaginary axis.  A correct implementation
    preserves this energy; a float64 cast zeroes it, leaving only the noise
    floor and producing no detections.

    Because a 90-degree rotation is a unit-magnitude phase shift, the
    correlation surface magnitude is unchanged.  This means code phase and
    Doppler estimates must be bit-for-bit identical to the real-signal
    baseline — any deviation indicates a bug in complex handling.
    """

    def test_imaginary_axis_signal_detects_all_svs(self, acq_fixture):
        """All SVs must be detected when signal energy is on the imaginary axis."""
        signal, codes, metadata, sv_truth = acq_fixture
        config = _make_config()

        # Rotate 90°: all signal energy moves to the imaginary axis.
        # With a float64 cast regression, real(j * signal) = 0 → no detection.
        signal_imag = signal.astype(np.complex128) * 1j

        results = acquire(signal_imag, codes, metadata, config)

        for sv in sv_truth:
            assert results[sv.prn].detected, (
                f"PRN {sv.prn} not detected with signal on imaginary axis "
                f"(peak_metric={results[sv.prn].peak_metric:.3f})"
            )

    def test_imaginary_axis_signal_matches_real_baseline(self, acq_fixture):
        """
        Code phase and Doppler from imaginary-axis signal must be identical
        to the real-signal baseline.  A 90-degree rotation does not change
        correlation magnitude, so peak locations are invariant.
        """
        signal, codes, metadata, sv_truth = acq_fixture
        config = _make_config()

        results_real = acquire(signal, codes, metadata, config)
        results_imag = acquire(signal.astype(np.complex128) * 1j, codes, metadata, config)

        for sv in sv_truth:
            assert (
                results_real[sv.prn].code_phase_samples
                == results_imag[sv.prn].code_phase_samples
            ), (
                f"PRN {sv.prn}: code phase {results_imag[sv.prn].code_phase_samples}"
                f" (imag axis) != {results_real[sv.prn].code_phase_samples} (real)"
            )
            assert results_real[sv.prn].doppler_hz == pytest.approx(
                results_imag[sv.prn].doppler_hz
            ), (
                f"PRN {sv.prn}: doppler {results_imag[sv.prn].doppler_hz:.1f} Hz"
                f" (imag axis) != {results_real[sv.prn].doppler_hz:.1f} Hz (real)"
            )