"""Policy-side import shim for the shared UniVTAC tactile encoder.

Keeping one implementation avoids drift between encoder pretraining and ACT
policy construction.  The default ``encoder_type='original'`` preserves the
legacy ResNet state-dict layout.
"""

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from encoder.network import *  # noqa: F401,F403,E402
