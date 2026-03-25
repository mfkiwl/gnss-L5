"""
GPS L5 PRN code generator.

Implements the L5 code generation algorithm per IS-GPS-705J Section 3.3.2.2.

Algorithm summary
-----------------
Each L5 code (I5 or Q5) is the modulo-2 sum (XOR) of two 13-stage LFSR sequences:

    XA: polynomial 1 + x^9 + x^10 + x^12 + x^13
        Initial state: all 1s.
        Short-cycled: resets to all 1s when the shift register reaches code
        vector 1111111111101 (after 8190 chips), producing a 10230-chip sequence
        from two partial XA periods.

    XB: polynomial 1 + x + x^3 + x^4 + x^6 + x^7 + x^8 + x^12 + x^13
        Initial state: PRN- and channel-specific, from Table 3-Ia / 3-Ib.
        Natural 8191-chip cycle; restarts from initial state at completion.

Code vector convention (IS-GPS-705J Section 3.3.2.2)
-----------------------------------------------------
State is expressed as a 13-character binary string where:
  - Rightmost character = stage 13 (current output, first bit out).
  - Remaining characters from right to left = stages 12 down to 1.
  - Equivalently, reading left to right: s12, s11, ..., s1, s13.

Internally, state is stored as a list where index 0 = stage 1 (s1) and
index 12 = stage 13 (s13). Stage 13 provides the output at each clock.
"""

import numpy as np

# ---------------------------------------------------------------------------
# XB initial code state table
# Source: IS-GPS-705J Tables 3-Ia and 3-Ib.
# Format: {prn: (i5_code_vector_str, q5_code_vector_str)}
# Each string is 13 binary characters in ICD code vector notation
# (rightmost bit = stage 13 = first bit out).
# ---------------------------------------------------------------------------

_XB_INITIAL_STATES: dict[int, tuple[str, str]] = {
    #      I5 code vector    Q5 code vector
    1:  ("0101011100100", "1001011001100"),
    2:  ("1100000110101", "0100011110110"),
    3:  ("0100000001000", "1111000100011"),
    4:  ("1011000100110", "0011101101010"),
    5:  ("1110111010111", "0011110110010"),
    6:  ("0110011111010", "0101010101001"),
    7:  ("1010010011111", "1111110000001"),
    8:  ("1011110100100", "0110101101000"),
    9:  ("1111100101011", "1011101000011"),
    10: ("0111111011110", "0010010000110"),
    11: ("0000100111010", "0001000000101"),
    12: ("1110011111001", "0101011000101"),
    13: ("0001110011100", "0100110100101"),
    14: ("0100000100111", "1010000111111"),
    15: ("0110101011010", "1011110001111"),
    16: ("0001111001001", "1101001011111"),
    17: ("0100110001111", "1110011001000"),
    18: ("1111000011110", "1011011100100"),
    19: ("1100100011111", "0011001011011"),
    20: ("0110101101101", "1100001110001"),
    21: ("0010000001000", "0110110010000"),
    22: ("1110111101111", "0010110001110"),
    23: ("1000011111110", "1000101111101"),
    24: ("1100010110100", "0110111110011"),
    25: ("1101001101101", "0100010011011"),
    26: ("1010110010110", "0101010111100"),
    27: ("0101011011110", "1000011111010"),
    28: ("0111101010110", "1111101000010"),
    29: ("0101111100001", "0101000100100"),
    30: ("1000010110111", "1000001111001"),
    31: ("0001010011110", "0101111100101"),
    32: ("0000010111001", "1001000101010"),
    33: ("1101010000001", "1011001000100"),
    34: ("1101111111001", "1111001000100"),
    35: ("1111011011100", "0110010110011"),
    36: ("1001011001000", "0011110101111"),
    37: ("0011010010000", "0010011010001"),
    38: ("0101100000110", "1111110011101"),
    39: ("1001001100101", "0101010011111"),
    40: ("1100111001010", "1000110101010"),
    41: ("0111011011001", "0010111100100"),
    42: ("0011101101100", "1011000100000"),
    43: ("0011011111010", "0011001011001"),
    44: ("1001011010001", "1000100101000"),
    45: ("1001010111111", "0000001111110"),
    46: ("0111000111101", "0000000010011"),
    47: ("0000001000100", "0101110011110"),
    48: ("1000101010001", "0001001000111"),
    49: ("0011010001001", "0011110000100"),
    50: ("1000111110001", "0100101011100"),
    51: ("1011100101001", "0010100011111"),
    52: ("0100101011010", "1101110011001"),
    53: ("0000001000010", "0011111101111"),
    54: ("0110001101110", "1100100110111"),
    55: ("0000011001110", "1001001100110"),
    56: ("1110111011110", "0100010011001"),
    57: ("0001000010011", "0000000001011"),
    58: ("0000010100001", "0000001101111"),
    59: ("0100001100001", "0101101101111"),
    60: ("0100101001001", "0100100001101"),
    61: ("0011110011110", "1101100101011"),
    62: ("1011000110001", "1010111000100"),
    63: ("0101111001011", "0010001101001"),
}

# XA reset detection state in internal [s1..s13] representation.
# Corresponds to code vector 1111111111101 (IS-GPS-705J Figure 3-2):
#   s1=0, s2..s13=1.
_XA_RESET_STATE: list[int] = [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _parse_code_vector(cv: str) -> list[int]:
    """Convert an ICD code vector string to internal state list [s1..s13].

    ICD code vector format (IS-GPS-705J Section 3.3.2.2):
      - Position 0 (leftmost) = s12
      - Positions 1..11 = s11 down to s1
      - Position 12 (rightmost) = s13

    Internal format: state[i] = s(i+1), so state[0]=s1, state[12]=s13.

    Parameters
    ----------
    cv : str
        13-character binary string in ICD code vector notation.

    Returns
    -------
    list[int]
        State as list of 13 integers (0 or 1), index 0 = s1, index 12 = s13.
    """
    assert len(cv) == 13, f"Code vector must be 13 characters, got {len(cv)}"
    # Positions 11..0 in cv give s1..s12; position 12 gives s13.
    return [int(cv[11 - i]) for i in range(12)] + [int(cv[12])]


def _generate_xa() -> np.ndarray:
    """Generate the 10230-chip XA sequence.

    Polynomial: 1 + x^9 + x^10 + x^12 + x^13
    Feedback taps: stages 9, 10, 12, 13 (indices 8, 9, 11, 12).
    Initial state: all 1s.
    Short-cycled: on detecting code vector 1111111111101, resets to all 1s
    on the following clock (IS-GPS-705J Figure 3-2).

    Returns
    -------
    np.ndarray
        Shape (10230,), dtype uint8, values 0 or 1.
    """
    state = [1] * 13  # set initial state to all 1s
    chips = np.zeros(10230, dtype=np.uint8)

    for i in range(10230):
        chips[i] = state[12]                          # stage 13 = current output

        if state == _XA_RESET_STATE:
            state = [1] * 13                          # short-cycle reset
        else:
            fb = state[8] ^ state[9] ^ state[11] ^ state[12]
            state = [fb] + state[:12]                 # shift toward higher stages

    return chips


def _generate_xb(initial_state: list[int]) -> np.ndarray:
    """Generate the 10230-chip XB sequence from a given initial state.

    Polynomial: 1 + x + x^3 + x^4 + x^6 + x^7 + x^8 + x^12 + x^13
    Feedback taps: stages 1, 3, 4, 6, 7, 8, 12, 13 (indices 0, 2, 3, 5, 6, 7, 11, 12).
    Natural length 8191 chips; restarts from initial state at completion.

    Parameters
    ----------
    initial_state : list[int]
        13-element state list [s1..s13] from _parse_code_vector().

    Returns
    -------
    np.ndarray
        Shape (10230,), dtype uint8, values 0 or 1.
    """
    state = list(initial_state)
    chips = np.zeros(10230, dtype=np.uint8)
    chip_count = 0

    for i in range(10230):
        chips[i] = state[12]                          # stage 13 = current output
        chip_count += 1

        fb = (state[0] ^ state[2]  ^ state[3]  ^ state[5] ^
              state[6] ^ state[7]  ^ state[11] ^ state[12])
        state = [fb] + state[:12]

        if chip_count == 8191:                        # natural period complete
            state = list(initial_state)
            chip_count = 0

    return chips


# Pre-compute XA once at module load — it is identical for every PRN.
_XA: np.ndarray = _generate_xa()


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------

class L5Code:
    """GPS L5 PRN code for a single satellite and channel.

    Generates the 10230-chip I5 or Q5 code sequence per IS-GPS-705J
    Section 3.3.2.2. The code is computed once at construction and stored
    as a binary chip array. The bipolar representation and sampled replica
    are derived on demand.

    Parameters
    ----------
    prn : int
        GPS PRN number (1–63).
    channel : str
        Channel identifier: ``"I5"`` (data channel) or ``"Q5"`` (pilot channel).

    Attributes
    ----------
    prn : int
        GPS PRN number.
    channel : str
        Channel identifier.
    chips : np.ndarray
        10230-chip binary code sequence, dtype uint8, values 0 or 1.

    Raises
    ------
    ValueError
        If ``prn`` is outside 1–63 or ``channel`` is not ``"I5"`` or ``"Q5"``.
    """

    CODE_LENGTH: int = 10230
    CHIP_RATE: float = 10.23e6          # chips per second
    VALID_CHANNELS: tuple[str, ...] = ("I5", "Q5")

    def __init__(self, prn: int, channel: str) -> None:
        if not (1 <= prn <= 63):
            raise ValueError(f"PRN must be between 1 and 63, got {prn}")
        if channel not in self.VALID_CHANNELS:
            raise ValueError(
                f"channel must be one of {self.VALID_CHANNELS}, got {channel!r}"
            )

        self.prn = prn
        self.channel = channel
        self.chips: np.ndarray = self._generate_chips()

    # ------------------------------------------------------------------
    # Private methods
    # ------------------------------------------------------------------

    def _generate_chips(self) -> np.ndarray:
        """Compute the 10230-chip binary code sequence."""
        cv_i5, cv_q5 = _XB_INITIAL_STATES[self.prn]
        cv = cv_i5 if self.channel == "I5" else cv_q5
        initial_state = _parse_code_vector(cv)
        xb = _generate_xb(initial_state)
        return (_XA ^ xb).astype(np.uint8)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def chips_bipolar(self) -> np.ndarray:
        """10230-chip bipolar code sequence, dtype int8, values +1 or -1.

        Mapping: binary 0 → +1, binary 1 → -1.
        Computed on demand from ``self.chips``; not stored separately.
        """
        return (1 - 2 * self.chips).astype(np.int8)

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def sample_replica(
        self,
        sample_rate: float,
        *,
        num_samples: int | None = None,
    ) -> np.ndarray:
        """Generate a sampled code replica at the given sample rate.

        Each output sample takes the value of the chip active at that
        sample instant (nearest-chip mapping via floor indexing).

        Parameters
        ----------
        sample_rate : float
            Receiver sample rate in Hz.
        num_samples : int, optional
            Number of output samples. Defaults to one full code period
            (``round(sample_rate / CHIP_RATE * CODE_LENGTH)``).

        Returns
        -------
        np.ndarray
            Sampled replica, dtype int8, values +1 or -1.
        """
        samples_per_period = round(sample_rate / self.CHIP_RATE * self.CODE_LENGTH)
        if num_samples is None:
            num_samples = samples_per_period

        chip_indices = np.floor(
            np.arange(num_samples) * self.CHIP_RATE / sample_rate
        ).astype(int)
        chip_indices = np.clip(chip_indices, 0, self.CODE_LENGTH - 1)
        return self.chips_bipolar[chip_indices]