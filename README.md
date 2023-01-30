# Evaluation Code for 3D Shape Generation

Earth Mover's Distance implementation is borrowed from [https://github.com/Colin97/MSN-Point-Cloud-Completion/tree/master/emd](https://github.com/Colin97/MSN-Point-Cloud-Completion/tree/master/emd).

## Setup
```shell
pip install -r requirements.txt
pip install -e .
```

##Get Started
```python3
from eval3d import emdModule, Evaluator

emd = emdModule()
evaluator = Evaluator()

x = torch.rand(2,2048,3).cuda()
y = torch.rand(2,2048,3).cuda()


dist, assignment = emd(x, y, 0.005, 50)
print(f"EMD: {dist}")

evaluator.compute_all_metrics(x, y)
```
