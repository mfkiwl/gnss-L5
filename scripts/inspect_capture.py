#!/usr/bin/env python3
"""
Write a diagnostic PNG for a GPS L5 capture file.

Edit the configuration block below, then run:

    python scripts/inspect_capture.py
"""

from pathlib import Path

import numpy as np

from gnss_common.capture.capture_file import CaptureFile
from gnss_common.capture.capture_metadata import CaptureMetadata
from gnss_common.capture.inspect_capture import inspect_capture

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CAPTURE_PATH = Path('/mnt/e/L5_capture/L5_IF20KHz_FS18MHz/L5_IF20KHz_FS18MHz.bin')
SAMPLE_RATE  = 18e6    # Hz
IF_FREQ      = 20e3    # Hz
DTYPE        = np.int8
IS_COMPLEX   = True

OUTPUT_PATH  = Path('capture_diagnostic.png')
NUM_MS       = 5.0     # ms of data to read
SKIP_MS      = 0.0     # ms to skip at the start of the file
# ---------------------------------------------------------------------------

capture = CaptureFile(CAPTURE_PATH, dtype=DTYPE, is_complex=IS_COMPLEX)
metadata = CaptureMetadata(
    sample_rate=SAMPLE_RATE,
    center_freq=IF_FREQ,
    dtype=np.dtype(DTYPE),
    is_complex=IS_COMPLEX,
)

inspect_capture(capture, metadata, OUTPUT_PATH, num_ms=NUM_MS, skip_ms=SKIP_MS)
print(f'Wrote {OUTPUT_PATH.resolve()}')