# Evaluation Code for 3D Shape Generation

Earth Mover's Distance implementation is borrowed from [https://github.com/Colin97/MSN-Point-Cloud-Completion/tree/master/emd](https://github.com/Colin97/MSN-Point-Cloud-Completion/tree/master/emd).

## Setup
```shell
pip install -r requirements.txt
pip install -e .
```

## Get Started
```python3
from eval3d import emdModule, Evaluator

"""
Two point clouds should have same size and be normalized to [0,1].
The number of points should be a multiple of 1024.
The batch size should be no greater than 512.
"""
dummy_gt = torch.randn([32, 1024, 3])  
dummy_pred = torch.randn([32, 1024, 3])

evaluator = Evaluator(dummy_gt, dummy_pred, 128, device="cuda:0", metric="emd")
evaluator.compute_all_metrics(verbose=True)

evaluator = Evaluator(dummy_gt, dummy_pred, 128, device="cuda:0", metric="l2")
evaluator.compute_all_metrics(verbose=True, compute_jsd_together=False)
```
