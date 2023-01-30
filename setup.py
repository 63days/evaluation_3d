from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


setup(
    name="eval3d",
    version=0.0,
    packages=find_packages(),
)
