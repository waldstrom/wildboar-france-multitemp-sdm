# =============================================================================
# Project Title: Wild Boar Distribution Model for France
# Path: scripts/__init__.py
# Purpose: Package initialisation and configuration for the Python workflow.
# Process Step: Sets up logging and pandas display options when imported.
# Created by Adrian Meyer, Institute Geomatics, FHNW (adrian.meyer@fhnw.ch)
# Created: July 2025
# Version: v.0.1.0
# =============================================================================
"""
Wild-boar habitat-suitability - Python workflow package.
Importing this module sets a global pandas display option and the root logger.
"""
from pathlib import Path
import logging
import sys
import pandas as pd

# -----------------------------------------------------------------------------
# Set pandas precision & printing for interactive sessions
pd.set_option("display.precision", 4)

# -----------------------------------------------------------------------------
# Root logger - propagated to all sub-modules
import os

LOGS_DIR = Path(os.environ.get("LOGS_DIR", "outputs/logs"))
LOGS_DIR.mkdir(parents=True, exist_ok=True)
log_file = LOGS_DIR / "pipeline_{time}.log".format(time=pd.Timestamp.now().strftime("%Y%m%d_%H%M%S"))

log_level = logging.DEBUG if os.environ.get("DEBUG") == "1" else logging.INFO
logging.basicConfig(
    level=log_level,
    format="%(asctime)s [%(levelname)s] %(name)s > %(message)s",
    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(log_file, encoding="utf-8")],
)
logging.captureWarnings(True)   # Redirect warnings to the logging system
