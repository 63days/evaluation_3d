# CUDA-Level 3D Evaluation Script
The code computes MMD, COV and 1-NNA based on Chamfer Distance (CD) and Earth Mover's Distance (EMD).

Tested with torch < 2.0.0 versions.

## Compile
```
cd PyTorchEMD
python setup.py install
```

## Run
```
python run_eval.py --fake_dir {} --ref_dir {} --output_dir {OPTIONAL} --normalize {"unit_cube", "unit_sphere", "none"}
```

## Reference
The code is heavily borrowed from [https://github.com/alexzhou907/PVD/tree/main/metrics/](https://github.com/alexzhou907/PVD/tree/main/metrics/).
