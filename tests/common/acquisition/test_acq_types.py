import numpy as np
import pytest
from gnss_common.acquisition.acq_types import AcqConfig, AcqResult


# ---------------------------------------------------------------------------
# AcqConfig
# ---------------------------------------------------------------------------

class TestAcqConfig:

    def test_construction_with_valid_parameters(self):
        cfg = AcqConfig(
            doppler_search_range_hz=5000.0,
            coh_int_ms=1,
            non_coh_count=4,
            acq_threshold=1.5,
        )
        assert cfg.doppler_search_range_hz == 5000.0
        assert cfg.coh_int_ms == 1
        assert cfg.non_coh_count == 4
        assert cfg.acq_threshold == 1.5

    def test_return_correlation_surface_defaults_false(self):
        cfg = AcqConfig(
            doppler_search_range_hz=5000.0,
            coh_int_ms=1,
            non_coh_count=4,
            acq_threshold=1.5,
        )
        assert cfg.return_correlation_surface is False

    def test_return_correlation_surface_can_be_set_true(self):
        cfg = AcqConfig(
            doppler_search_range_hz=5000.0,
            coh_int_ms=1,
            non_coh_count=4,
            acq_threshold=1.5,
            return_correlation_surface=True,
        )
        assert cfg.return_correlation_surface is True

    def test_coh_int_ms_must_be_positive(self):
        with pytest.raises(ValueError):
            AcqConfig(
                doppler_search_range_hz=5000.0,
                coh_int_ms=0,
                non_coh_count=4,
                acq_threshold=1.5,
            )

    def test_coh_int_ms_must_be_integer(self):
        with pytest.raises((ValueError, TypeError)):
            AcqConfig(
                doppler_search_range_hz=5000.0,
                coh_int_ms=1.5,
                non_coh_count=4,
                acq_threshold=1.5,
            )

    def test_non_coh_count_must_be_positive(self):
        with pytest.raises(ValueError):
            AcqConfig(
                doppler_search_range_hz=5000.0,
                coh_int_ms=1,
                non_coh_count=0,
                acq_threshold=1.5,
            )

    def test_doppler_search_range_must_be_positive(self):
        with pytest.raises(ValueError):
            AcqConfig(
                doppler_search_range_hz=0.0,
                coh_int_ms=1,
                non_coh_count=4,
                acq_threshold=1.5,
            )

    def test_acq_threshold_must_be_positive(self):
        with pytest.raises(ValueError):
            AcqConfig(
                doppler_search_range_hz=5000.0,
                coh_int_ms=1,
                non_coh_count=4,
                acq_threshold=0.0,
            )

    def test_coh_int_ms_negative_raises(self):
        with pytest.raises(ValueError):
            AcqConfig(
                doppler_search_range_hz=5000.0,
                coh_int_ms=-1,
                non_coh_count=4,
                acq_threshold=1.5,
            )

    def test_residual_if_hz_is_stored(self):
        cfg = AcqConfig(
            doppler_search_range_hz=5000.0,
            coh_int_ms=1,
            non_coh_count=4,
            acq_threshold=1.5,
            residual_if_hz=1234.0,
        )
        assert cfg.residual_if_hz == pytest.approx(1234.0)

    def test_residual_if_hz_defaults_to_zero(self):
        cfg = AcqConfig(
            doppler_search_range_hz=5000.0,
            coh_int_ms=1,
            non_coh_count=4,
            acq_threshold=1.5,
        )
        assert cfg.residual_if_hz == 0.0


# ---------------------------------------------------------------------------
# AcqResult
# ---------------------------------------------------------------------------

class TestAcqResult:

    def test_construction_detected_signal(self):
        result = AcqResult(
            code_phase_samples=1000,
            doppler_hz=-2000.0,
            doppler_phase_rad=1.23,
            peak_metric=2.1,
            detected=True,
        )
        assert result.code_phase_samples == 1000
        assert result.doppler_hz == -2000.0
        assert result.doppler_phase_rad == pytest.approx(1.23)
        assert result.peak_metric == pytest.approx(2.1)
        assert result.detected is True

    def test_correlation_surface_defaults_none(self):
        result = AcqResult(
            code_phase_samples=1000,
            doppler_hz=-2000.0,
            doppler_phase_rad=1.23,
            peak_metric=2.1,
            detected=True,
        )
        assert result.correlation_surface is None

    def test_correlation_surface_can_be_set(self):
        surface = np.zeros((11, 20460))
        result = AcqResult(
            code_phase_samples=1000,
            doppler_hz=-2000.0,
            doppler_phase_rad=1.23,
            peak_metric=2.1,
            detected=True,
            correlation_surface=surface,
        )
        assert result.correlation_surface is surface

    def test_construction_not_detected(self):
        result = AcqResult(
            code_phase_samples=0,
            doppler_hz=0.0,
            doppler_phase_rad=0.0,
            peak_metric=1.1,
            detected=False,
        )
        assert result.detected is False