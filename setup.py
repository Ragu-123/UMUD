from setuptools import setup, find_packages

setup(
    name="umud",
    version="1.0.0",
    description="UMUD Muscle Architecture Ultrasound Metrology & Deep Learning Pipeline",
    author="UMUD Autonomous Agent Team",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "timm>=0.9.0",
        "albumentations>=1.3.0",
        "opencv-python-headless>=4.7.0",
        "scipy>=1.10.0",
        "numpy>=1.23.0",
        "pandas>=1.5.0",
        "scikit-learn>=1.2.0",
        "tqdm>=4.65.0",
    ],
)
