from setuptools import setup, find_packages

setup(
    name='numpy_quazi_torch',
    version="0.0.1",
    author="Grga",
    description=("Pytorch... but numpy."),
    packages=find_packages(
        include=[
            "numpy_quazi_torch",
            "numpy_quazi_torch.*",
            "numpy_quazi_torch.custom_models",
            "numpy_quazi_torch.custom_models.*",
            "numpy_quazi_torch.data",
            "numpy_quazi_torch.data.*",
            "numpy_quazi_torch.models",
            "numpy_quazi_torch.models.*",
            "numpy_quazi_torch.scheduler",
            "numpy_quazi_torch.scheduler.*",
            "numpy_quazi_torch.scirpts",
            "numpy_quazi_torch.scripts.*",
        ]
    ),
    install_requires=[
        "numpy",
        "scipy",
        "scikit-learn",
        "streamlit",
        "tqdm",
        "scikit-image",
        "opencv-python",
        "matplotlib"
    ],
    entry_points = {
        'console_scripts': [
            "train=numpy_quazi_torch.scripts.train:main"
        ]
    }
)