from setuptools import setup, find_packages

setup(
    name='human-policy',
    version='0.0.1',
    packages=find_packages(),
    description='Humanoid Policy ~ Human Policy',
    author='UCSD Xiaolong Wang Group',
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