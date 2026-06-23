import numpy as np
from skimage.data import shepp_logan_phantom, camera
from skimage.transform import resize
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DATA_DIR, SHEPP_LOGAN_PATH, CAMERA_PATH, IMAGE_SIZE

def generate_all():
    DATA_DIR.mkdir(exist_ok = True)

    #Shepp Logan canonical benchmark
    s1 = shepp_logan_phantom()      #Original is 400x400
    s1 = resize(s1, (IMAGE_SIZE, IMAGE_SIZE))       #Resize to reduce image size 
    s1 = s1/s1.max()        #Normalize to [0, 1]
    np.save(SHEPP_LOGAN_PATH, s1)
    print (f"Saved Shepp Logan phantom:\nShape: {s1.shape}\nPath: {SHEPP_LOGAN_PATH}")

    #Camera as natural benchamark
    cam = camera().astype(float)
    cam = resize(cam, (IMAGE_SIZE, IMAGE_SIZE))
    cam = cam/cam.max()
    np.save(CAMERA_PATH, cam)
    print (f"Saved camera image:\nShape: {cam.shape}\nPath: {CAMERA_PATH}")

if __name__ == "__main__":         
    generate_all()                 
    """
    This ensure that this file only executes when run from terminal
    Doing so wont allow "from data.generate import generate_all"  
    and therefore prevent accidentally generating files and overwriting them.

    -------------------------------------------------------------------------------

    To verify if this file has executed successfully, check in terminal and type: 
    python data/generate.py
    """

