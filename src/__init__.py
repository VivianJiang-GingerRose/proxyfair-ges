"""Repository source namespace and bundled dependency routing."""

from pathlib import Path
import sys


# The bundled causal-learn fork retains upstream absolute ``causallearn.*``
# imports. Route those imports to the fork before third-party libraries can
# import a separately installed causal-learn release into ``sys.modules``.
_BUNDLED_CAUSAL_LEARN = str(Path(__file__).resolve().parent / "faircausal" / "causal_learn")
if _BUNDLED_CAUSAL_LEARN in sys.path:
    sys.path.remove(_BUNDLED_CAUSAL_LEARN)
sys.path.insert(0, _BUNDLED_CAUSAL_LEARN)
