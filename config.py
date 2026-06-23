from pathlib import Path

#Repo root is where the `config.py` file is located
ROOT = Path(__file__).parent

DATA_DIR = ROOT/'data'
SRC_DIR = ROOT/'src'
RESULTS_DIR = ROOT/'results'

#Canonical data files
SHEPP_LOGAN_PATH = DATA_DIR / "shepp_logan.npy"
CAMERA_PATH = DATA_DIR / "camera.npy"

#Experiement parameters
IMAGE_SIZE = 128         #Size of unknown matrix  X = 128x128 pixels, where X belongs to R^(128x128)
N_MEASUREMENTS =  5000        #As y = A(X), y belongs to R^ N_MEASUREMENTS represents number of pixels observed. This works because we are assuming that X has low rank.
COMPRESSION_RATIO =  N_MEASUREMENTS/(IMAGE_SIZE**2)        

RANDOM_SEED = 42
RHO = 1.5     #ADMM penalty factor (do not take - 0.1, 0.5, 1.2, 1.8)
MAX_ITER = 1000
TOL = 1e-5   #Tolerance

"""
To verify if this file has executed successfully, check in terminal and type: 
python -c "from config import *; print(ROOT); print(SHEPP_LOGAN_PATH); print(IMAGE_SIZE)"
"""