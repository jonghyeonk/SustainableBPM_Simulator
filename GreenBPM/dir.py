# Copyright 2022. Jonghyeon Ko
#
# All scripts in this folder are distributed as a free software. 
# We extended a few functions for fundamental concepts of process mining introduced in the complementary repository for BINet [1].
# 
# 1. [Nolle, T., Seeliger, A., Mühlhäuser, M.: BINet: Multivariate Business Process Anomaly Detection Using Deep Learning, 2018](https://doi.org/10.1007/978-3-319-98648-7_16)
# ==============================================================================


from pathlib import Path

# Base
ROOT_DIR = Path(__file__).parent.parent

# Base directories
OUT_DIR = ROOT_DIR / '.out'  # For anything that is being generated
RES_DIR = ROOT_DIR / '.res'  # For resources shipped with the repository
IMG_DIR = ROOT_DIR / 'img'

# Resources
PROCESS_MODEL_DIR = RES_DIR / 'process_models'  # Randomly generated process models from PLG2
CSV_DIR = RES_DIR / 'csvdata'  
OUTPUT_CSV_DIR = OUT_DIR / 'csvdata'

# Output
PLOT_DIR = OUT_DIR / 'plots'  # For plots
TEMP_DIR = OUT_DIR / 'temp'

# Config
CONFIG_DIR = ROOT_DIR / '.config'

# Misc
DATE_FORMAT = 'YYYYMMDD-HHmmss.SSSSSS'


def generate():
    """Generate directories."""
    dirs = [
        ROOT_DIR,
        RES_DIR,
        PROCESS_MODEL_DIR,
        CSV_DIR,
        PLOT_DIR
    ]
    for d in dirs:
        if not d.exists():
            d.mkdir()


def get_process_model_files(path=None):
    if path is None:
        path = PROCESS_MODEL_DIR
    for f in path.glob('*.plg'):
        yield f.stem

