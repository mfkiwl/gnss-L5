#!/usr/bin/env python3
"""
Compare GPS L5 I5 codes between the Octave reference implementation and
the Python L5Code generator.

Reads the binary file produced by write_l5i_codes.m and compares each PRN
chip-by-chip against L5Code(prn, 'I5').chips.

Usage:
    python scripts/compare_l5_codes.py [path/to/l5i_codes_octave.bin]
"""

import sys
from pathlib import Path

import numpy as np

from gnss_l5.codes.l5_code import L5Code

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DEFAULT_BIN = Path('l5i_codes_octave.bin')
NUM_PRNS    = 32
CODE_LENGTH = 10230
# ---------------------------------------------------------------------------

bin_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_BIN

if not bin_path.exists():
    print(f'Error: file not found: {bin_path}')
    sys.exit(1)

raw = np.frombuffer(bin_path.read_bytes(), dtype=np.uint8)

expected_bytes = NUM_PRNS * CODE_LENGTH
if len(raw) != expected_bytes:
    print(f'Error: expected {expected_bytes} bytes, got {len(raw)}')
    sys.exit(1)

octave_codes = raw.reshape(NUM_PRNS, CODE_LENGTH)

# ---------------------------------------------------------------------------
# Compare PRN by PRN
# ---------------------------------------------------------------------------
all_match = True

for i in range(NUM_PRNS):
    prn = i + 1
    python_chips  = L5Code(prn, 'I5').chips          # uint8, values 0/1
    octave_chips  = octave_codes[i]                  # uint8, values 0/1

    mismatches = int(np.sum(python_chips != octave_chips))

    if mismatches == 0:
        print(f'PRN {prn:2d}:  OK')
    else:
        all_match = False
        first_idx = int(np.argmax(python_chips != octave_chips))
        print(
            f'PRN {prn:2d}:  MISMATCH — {mismatches} chip(s) differ, '
            f'first at chip {first_idx} '
            f'(Python={python_chips[first_idx]}, Octave={octave_chips[first_idx]})'
        )

print()
if all_match:
    print('All 32 codes match.')
else:
    print('One or more codes do not match — check output above.')
    sys.exit(1)