from setuptools import setup, find_packages

setup(
    name='cross-embodiment-transformer',
    version='0.0.1',
    packages=find_packages(),
    description='Cross Embodiment Transformer for Human2Robot',
    author='CXX, LJL, etc.',
    install_requires=[
        'numpy',
        'opencv-python',
        'h5py',
        'pyzed',
        'tqdm',
        'torch',
        'matplotlib'
    ],
) 