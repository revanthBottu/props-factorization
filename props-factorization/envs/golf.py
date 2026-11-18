"""Register the sibling golf simulation as a Gymnasium environment for ProPS."""

from pathlib import Path
import sys

from dotenv import load_dotenv


SIMULATION_DIR = Path(__file__).resolve().parents[2] / "simulationsetup"
if str(SIMULATION_DIR) not in sys.path:
    sys.path.insert(0, str(SIMULATION_DIR))

# The golf runner also reads this file. Shell variables take precedence.
load_dotenv(SIMULATION_DIR / ".env", override=False)

# Importing the module registers GolfSwing-v0 with Gymnasium.
import golf_rl_env  # noqa: E402,F401
