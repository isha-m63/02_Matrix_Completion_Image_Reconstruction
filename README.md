# Repository structure

```text
Matrix_Completion_Image_Reconstruction/
│
├── README.md
│
├── data/
│   ├── generate.py                 #Generate, resize and normalize 
│
├── src/
│   ├── __init__.py
│   ├── operators.py               #Measurement operator A and adjoint A*, update Z for ADMM optimizer
│   ├── admm.py                    #ADMM solver
│   ├── svt.py                     #Singular value thresholding
│   └── metrics.py                 #PSNR, SSIM, relative error
│
├── experiments/
|   ├── 01_admm_convergence_simulated_data.ipynb
|   ├── 02_admm_convergence_shepp_logan.ipynb
|   ├── 03_generalisation_camera_image.ipynb
│
└── tests/
    ├── test_svt.py                 #Verify SVT against known closed form
    ├── test_operators.py           #Verify A* is true adjoint of A
    └── test_admm.py               #Verify solver converges on toy problem
    └── test_metrics.py            #Verify the metrics on toy problem

