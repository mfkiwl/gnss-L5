"""L5 PRN code verification: autocorrelation and cross-correlation properties.

Sanity-checks the GPS L5 code generator beyond the per-chip ICD test
fixtures by examining bulk correlation statistics.

Three checks
------------
1. Autocorrelation peak and max sidelobe for all 63 PRNs * {I5, Q5}.
   The peak at zero lag MUST equal CODE_LENGTH (10230); this is the only
   hard pass/fail criterion.
2. Same-PRN cross-correlation (I5 vs Q5) for all 63 PRNs.
3. Cross-correlation for NUM_RANDOM_PAIRS random pairs of distinct
   (prn, channel) codes.

Tests 2 and 3 only print statistics; there is no clean closed-form bound
for L5 (it is Gold-like but not strictly Gold), so interpretation is left
to the reader. As a rough sanity check, peak/sidelobe ratios in the tens
or better indicate the codes have correct correlation structure.

Usage
-----
Run from the project root with the editable package installed:

    python l5_code_verify.py

Exit code 0 on success, 1 if any autocorrelation peak is wrong.
"""

import numpy as np

from gnss_l5.codes.l5_code import L5Code

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

RNG_SEED = 42                                         # reproducible pair selection
NUM_RANDOM_PAIRS = 64
PRN_RANGE = range(1, 64)
CHANNELS = ("I5", "Q5")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def circular_correlation(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return the circular cross-correlation of two equal-length real sequences.

    Computed via FFT: r = IFFT(FFT(a) * conj(FFT(b))). For our purposes only
    magnitudes/peaks matter, so the exact sign convention on the lag index
    is unimportant.

    For a bipolar (+/-1) sequence of length N, the autocorrelation at zero
    lag is exactly N.
    """
    return np.real(np.fft.ifft(np.fft.fft(a) * np.conj(np.fft.fft(b))))


def build_all_codes() -> dict[tuple[int, str], np.ndarray]:
    """Generate every L5 code as a float64 bipolar array, keyed by (prn, channel)."""
    return {
        (prn, ch): L5Code(prn, ch).chips_bipolar.astype(np.float64)
        for prn in PRN_RANGE
        for ch in CHANNELS
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def check_autocorrelation(codes: dict[tuple[int, str], np.ndarray]) -> bool:
    """Verify peak == CODE_LENGTH for every code; report max sidelobe stats.

    Returns True on success, False if any peak is wrong.
    """
    print("=" * 64)
    print("Test 1: Autocorrelation (all 63 PRNs, both channels)")
    print("=" * 64)

    expected_peak = L5Code.CODE_LENGTH
    peak_failures: list[str] = []
    sidelobes: list[float] = []
    worst_offender: tuple[int, str, float] | None = None

    for (prn, ch), code in codes.items():
        ac = circular_correlation(code, code)
        peak = ac[0]
        max_sl = float(np.abs(ac[1:]).max())
        sidelobes.append(max_sl)

        if not np.isclose(peak, expected_peak):
            peak_failures.append(
                f"  PRN {prn:>2} {ch}: peak = {peak:.1f}, expected {expected_peak}"
            )
        if worst_offender is None or max_sl > worst_offender[2]:
            worst_offender = (prn, ch, max_sl)

    print(f"Codes tested:           {len(codes)}")
    print(f"Expected peak:          {expected_peak}")
    print(f"Peak failures:          {len(peak_failures)}")
    for line in peak_failures:
        print(line)
    print(f"Max sidelobe (worst):   {max(sidelobes):.0f}  "
          f"[PRN {worst_offender[0]} {worst_offender[1]}]")
    print(f"Max sidelobe (mean):    {np.mean(sidelobes):.1f}")
    print(f"Peak / worst sidelobe:  {expected_peak / max(sidelobes):.1f}")
    print()
    return len(peak_failures) == 0


def check_same_prn_cross_correlation(
    codes: dict[tuple[int, str], np.ndarray],
) -> None:
    """Report I5 vs Q5 cross-correlation statistics for every PRN."""
    print("=" * 64)
    print("Test 2: Same-PRN cross-correlation (I5 vs Q5)")
    print("=" * 64)

    max_xcs: list[float] = []
    worst: tuple[int, float] | None = None
    for prn in PRN_RANGE:
        i5 = codes[(prn, "I5")]
        q5 = codes[(prn, "Q5")]
        xc = circular_correlation(i5, q5)
        max_xc = float(np.abs(xc).max())
        max_xcs.append(max_xc)
        if worst is None or max_xc > worst[1]:
            worst = (prn, max_xc)

    print(f"PRNs tested:                  63")
    print(f"Code length (peak ref):       {L5Code.CODE_LENGTH}")
    print(f"Max cross-correlation:        {max(max_xcs):.0f}  [PRN {worst[0]}]")
    print(f"Mean max cross-correlation:   {np.mean(max_xcs):.1f}")
    print(f"Peak / worst cross-corr:      {L5Code.CODE_LENGTH / max(max_xcs):.1f}")
    print()


def check_random_cross_correlation(
    codes: dict[tuple[int, str], np.ndarray],
) -> None:
    """Report cross-correlation for NUM_RANDOM_PAIRS random distinct-code pairs."""
    print("=" * 64)
    print(f"Test 3: Random cross-correlation ({NUM_RANDOM_PAIRS} pairs, "
          f"seed={RNG_SEED})")
    print("=" * 64)

    rng = np.random.default_rng(RNG_SEED)
    keys = list(codes.keys())
    pairs: list[tuple[tuple[int, str], tuple[int, str]]] = []
    while len(pairs) < NUM_RANDOM_PAIRS:
        i, j = rng.choice(len(keys), size=2, replace=False)
        pairs.append((keys[i], keys[j]))

    print(f"{'PRN_A':>5} {'CH_A':>5}   {'PRN_B':>5} {'CH_B':>5}   {'Max |XC|':>9}")
    print("-" * 47)

    max_xcs: list[float] = []
    for (prn_a, ch_a), (prn_b, ch_b) in pairs:
        xc = circular_correlation(codes[(prn_a, ch_a)], codes[(prn_b, ch_b)])
        max_xc = float(np.abs(xc).max())
        max_xcs.append(max_xc)
        print(f"{prn_a:>5} {ch_a:>5}   {prn_b:>5} {ch_b:>5}   {max_xc:>9.0f}")

    print("-" * 47)
    print(f"Max:    {max(max_xcs):.0f}")
    print(f"Mean:   {np.mean(max_xcs):.1f}")
    print(f"Peak / worst:  {L5Code.CODE_LENGTH / max(max_xcs):.1f}")
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    codes = build_all_codes()
    autocorr_ok = check_autocorrelation(codes)
    check_same_prn_cross_correlation(codes)
    check_random_cross_correlation(codes)

    print("=" * 64)
    print("Verification " + ("PASSED" if autocorr_ok else "FAILED"))
    print("=" * 64)
    return 0 if autocorr_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())