from setuptools import find_packages, setup

setup(
    name="dlordinal",
    version="0.1",
    description="Deep learning for ordinal classification",
    author="Francisco Bérchez, Víctor Vargas",
    author_email="i72bemof@uco.es, vvargas@uco.es",
    license="Universidad de Córdoba",
    packages=find_packages(),
    install_requires=[
        "scikit-learn",
        "numpy",
        "torch==1.10.2",
        "torchvision==0.11.3",
        "pandas",
        "scipy",
        "matplotlib",
        "seaborn",
        "scikit-image",
        "tqdm",
        "Pillow",
        "pytest",
    ],
    zip_safe=False,
)
