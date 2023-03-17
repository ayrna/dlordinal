from setuptools import setup, find_packages

setup(name='dlordinal',
      version='0.1.6',
      description='Deep learning for ordinal classification',
      author='Francisco Bérchez, Víctor Vargas',
      author_email='i72bemof@uco.es, vvargas@uco.es',
      license='Universidad de Córdoba',
      packages=find_packages(),
      install_requires=['sklearn', 'numpy', 'torch', 'torchvision', 'pandas', 'scipy', 'matplotlib', 'seaborn', 'bioinfokit', 'statsmodels', 'scikit-posthocs', 'typing', 'itertools', 'abc', 're', 'skimage', 'tqdm'],
      zip_safe=False)
