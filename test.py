from PyTorchEMD.emd import earth_mover_distance as EMD
import time
import torch

class EMD_Wrapper(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, y):
        return EMD(x, y, transpose=False)


x= torch.rand(100,2048,3).cuda()
y=torch.rand(100,2048,3).cuda()

emd_module = EMD_Wrapper()
# emd_module = torch.nn.DataParallel(emd_module)

start =time.time()
d = emd_module(x,y)
end = time.time()
print(f"{end-start:.6f}s")
