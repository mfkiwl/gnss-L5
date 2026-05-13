# tests/codes/test_l5_code.py
"""
Tests for GPS L5 PRN code generator (L5Code class).

Ground truth values from IS-GPS-705 Table 3-Ia.
First 10 chips for I5 and Q5 are given in binary in the ICD.
XB code advances are given in octal in the ICD.
"""

import numpy as np
import pytest
from gnss_l5.codes.l5_code import L5Code

# ---------------------------------------------------------------------------
# ICD ground truth fixtures
# First 10 chips from IS-GPS-705 Table 3-Ia, expressed as lists of 0/1.
# The chip sequence from an initial state reads out stage 13 first (the
# rightmost character of the code vector string), then on each successive
# clock the register shifts so that stage 12 becomes stage 13, and so on.
# Reading the code vector left to right gives stages s1..s12, s13 in
# ascending order.
# ---------------------------------------------------------------------------
 
ICD_FIRST_10_CHIPS = {
    (1, "I5"): [1, 1, 0, 1, 1, 0, 0, 0, 1, 0],
    (1, "Q5"): [1, 1, 0, 0, 1, 1, 0, 0, 1, 0],
    (2, "I5"): [0, 1, 0, 1, 0, 0, 1, 1, 1, 1],
    (2, "Q5"): [1, 0, 0, 1, 0, 0, 0, 0, 1, 1],
    (3, "I5"): [1, 1, 1, 0, 1, 1, 1, 1, 1, 1],
    (3, "Q5"): [0, 0, 1, 1, 1, 0, 1, 1, 1, 0],
}


# ---------------------------------------------------------------------------
# Group 1: Construction and attributes
# ---------------------------------------------------------------------------

class TestConstruction:

    def test_valid_i5_channel(self):
        code = L5Code(prn=1, channel="I5")
        assert code is not None

    def test_valid_q5_channel(self):
        code = L5Code(prn=1, channel="Q5")
        assert code is not None

    def test_prn_attribute(self):
        code = L5Code(prn=5, channel="I5")
        assert code.prn == 5

    def test_channel_attribute(self):
        code = L5Code(prn=5, channel="Q5")
        assert code.channel == "Q5"

    def test_invalid_prn_zero(self):
        with pytest.raises(ValueError):
            L5Code(prn=0, channel="I5")

    def test_invalid_prn_negative(self):
        with pytest.raises(ValueError):
            L5Code(prn=-1, channel="I5")

    def test_invalid_prn_too_large(self):
        with pytest.raises(ValueError):
            L5Code(prn=64, channel="I5")

    def test_invalid_channel_string(self):
        with pytest.raises(ValueError):
            L5Code(prn=1, channel="L5")

    def test_invalid_channel_case(self):
        """Channel matching must be exact — 'i5' is not 'I5'."""
        with pytest.raises(ValueError):
            L5Code(prn=1, channel="i5")


# ---------------------------------------------------------------------------
# Group 2: chips array
# ---------------------------------------------------------------------------

class TestChips:

    def test_chips_shape(self):
        code = L5Code(prn=1, channel="I5")
        assert code.chips.shape == (10230,)

    def test_chips_dtype(self):
        code = L5Code(prn=1, channel="I5")
        assert code.chips.dtype == np.uint8

    def test_chips_values_are_binary(self):
        code = L5Code(prn=1, channel="I5")
        assert set(np.unique(code.chips)).issubset({0, 1})


# ---------------------------------------------------------------------------
# Group 3: chips_bipolar property
# ---------------------------------------------------------------------------

class TestChipsBipolar:

    def test_chips_bipolar_shape(self):
        code = L5Code(prn=1, channel="I5")
        assert code.chips_bipolar.shape == (10230,)

    def test_chips_bipolar_dtype(self):
        code = L5Code(prn=1, channel="I5")
        assert code.chips_bipolar.dtype == np.int8

    def test_chips_bipolar_values_are_plus_minus_one(self):
        code = L5Code(prn=1, channel="I5")
        assert set(np.unique(code.chips_bipolar)).issubset({-1, 1})

    def test_chips_bipolar_mapping_zero_to_plus_one(self):
        """Binary 0 must map to +1."""
        code = L5Code(prn=1, channel="I5")
        zero_positions = code.chips == 0
        assert np.all(code.chips_bipolar[zero_positions] == 1)

    def test_chips_bipolar_mapping_one_to_minus_one(self):
        """Binary 1 must map to -1."""
        code = L5Code(prn=1, channel="I5")
        one_positions = code.chips == 1
        assert np.all(code.chips_bipolar[one_positions] == -1)

    def test_chips_bipolar_is_property_not_method(self):
        """chips_bipolar should be a @property, not a callable."""
        code = L5Code(prn=1, channel="I5")
        assert isinstance(code.chips_bipolar, np.ndarray)


# ---------------------------------------------------------------------------
# Group 4: ICD ground truth
# ---------------------------------------------------------------------------

class TestICDVerification:

    @pytest.mark.parametrize("prn,channel,expected_chips", [
        (prn_ch[0], prn_ch[1], chips)
        for prn_ch, chips in ICD_FIRST_10_CHIPS.items()
    ])
    def test_first_10_chips_match_icd(self, prn, channel, expected_chips):
        """First 10 chips must match IS-GPS-705 Table 3-Ia exactly."""
        code = L5Code(prn=prn, channel=channel)
        expected = np.array(expected_chips, dtype=np.uint8)
        np.testing.assert_array_equal(code.chips[:10], expected)


# ---------------------------------------------------------------------------
# Group 5: Code independence
# ---------------------------------------------------------------------------

class TestCodeIndependence:

    def test_i5_and_q5_differ_for_same_prn(self):
        i5 = L5Code(prn=1, channel="I5")
        q5 = L5Code(prn=1, channel="Q5")
        assert not np.array_equal(i5.chips, q5.chips)

    def test_different_prns_produce_different_i5_codes(self):
        code1 = L5Code(prn=1, channel="I5")
        code2 = L5Code(prn=2, channel="I5")
        assert not np.array_equal(code1.chips, code2.chips)

    def test_different_prns_produce_different_q5_codes(self):
        code1 = L5Code(prn=1, channel="Q5")
        code2 = L5Code(prn=2, channel="Q5")
        assert not np.array_equal(code1.chips, code2.chips)


# ---------------------------------------------------------------------------
# Group 6: sample_replica
# ---------------------------------------------------------------------------

class TestSampleReplica:

    CHIP_RATE = 10.23e6  # Hz

    def test_replica_at_chip_rate_has_length_10230(self):
        code = L5Code(prn=1, channel="I5")
        replica = code.sample_replica(sample_rate=self.CHIP_RATE)
        assert len(replica) == 10230

    def test_replica_at_chip_rate_equals_chips_bipolar(self):
        """At exactly the chip rate, one sample per chip."""
        code = L5Code(prn=1, channel="I5")
        replica = code.sample_replica(sample_rate=self.CHIP_RATE)
        np.testing.assert_array_equal(replica, code.chips_bipolar)

    def test_replica_at_2x_chip_rate_has_length_20460(self):
        code = L5Code(prn=1, channel="I5")
        replica = code.sample_replica(sample_rate=2 * self.CHIP_RATE)
        assert len(replica) == 20460

    def test_replica_at_2x_each_chip_is_repeated(self):
        """At 2x chip rate, each chip value appears twice consecutively."""
        code = L5Code(prn=1, channel="I5")
        replica = code.sample_replica(sample_rate=2 * self.CHIP_RATE)
        np.testing.assert_array_equal(replica[0::2], code.chips_bipolar)
        np.testing.assert_array_equal(replica[1::2], code.chips_bipolar)

    def test_replica_values_are_plus_minus_one(self):
        code = L5Code(prn=1, channel="I5")
        replica = code.sample_replica(sample_rate=2.048e7)
        assert set(np.unique(replica)).issubset({-1, 1})

    def test_replica_dtype_is_int8(self):
        code = L5Code(prn=1, channel="I5")
        replica = code.sample_replica(sample_rate=self.CHIP_RATE)
        assert replica.dtype == np.int8