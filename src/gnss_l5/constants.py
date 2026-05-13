"""
GPS L5 signal constants.

Physical and standard values that do not vary with hardware or capture
configuration.  Capture-specific values (sample rate, residual IF, etc.)
belong in CaptureMetadata or AcqConfig, not here.
"""

# ---------------------------------------------------------------------------
# L5 signal parameters  (IS-GPS-705J)
# ---------------------------------------------------------------------------

L5_NOMINAL_FREQ_HZ: float = 1176.45e6  # L5 carrier centre frequency, Hz
L5_CHIP_RATE_HZ:    float = 10.23e6    # chips per second
L5_CODE_LENGTH:     int   = 10230      # chips per code period (1 ms)

# ---------------------------------------------------------------------------
# L5-capable PRN list
#
# Source: https://en.wikipedia.org/wiki/List_of_GPS_satellites (April 2026)
#
# Only Block IIF (launched 2010–2016) and Block III (launched 2018–2026)
# satellites transmit L5.  Block IIR and IIR-M do not.
#
# Block IIF — 11 operational as of 2026:
#   PRNs 3, 6, 8, 9, 10, 24, 25, 26, 27, 30, 32
#   (SVN-63 / PRN 1 retired from L5 service July 2023; PRN 1 reassigned
#    to Block III SVN-80 in late 2024.)
#
# Block III (SV01–SV10) — 9 operational + 1 commissioning as of April 2026:
#   PRNs 1, 4, 11, 14, 18, 20, 21, 23, 28  (operational)
#   PRN  13                                  (commissioning, launched Apr 2026)
#
# Newer Block III PRNs (20, 21, 13) launched late 2025 / early 2026 and will
# not be present in captures predating those launches.  Searching for an
# unoccupied PRN is harmless — it simply produces no detection.
# ---------------------------------------------------------------------------

L5_CAPABLE_PRNS: tuple[int, ...] = (
    1, 3, 4, 6, 8, 9, 10, 11, 13, 14, 18, 20, 21, 23, 24, 25, 26, 27, 28, 30, 32
)