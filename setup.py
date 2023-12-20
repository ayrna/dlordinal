from setuptools import find_packages, setup

setup(
    name="dlordinal",
    version="0.1.0",
    description="Deep learning for ordinal classification",
    author="Francisco Bérchez, Víctor Vargas",
    author_email="i72bemof@uco.es, vvargas@uco.es",
    license="Universidad de Córdoba",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "scikit-learn==1.*",
        "numpy>=1.21,==1.*",
        "torch==2.*",
        "torchvision>=0.13",
        "pandas>=1",
        "scipy>=1.7",
        "matplotlib>=3.1",
        "seaborn>=0.12",
        "scikit-image>=0.18",
        "tqdm>=4",
        "Pillow>=8",
    ],
    extras_require={
        "dev": [
            "pytest",
            "pre-commit",
        ]
    },
    zip_safe=False,
)
