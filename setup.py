import re
from pathlib import Path

from setuptools import find_packages, setup

classes = """
    License :: OSI Approved :: MIT License
    Topic :: Software Development :: Libraries
    Topic :: Scientific/Engineering
    Topic :: Scientific/Engineering :: Bio-Informatics
    Programming Language :: Python :: 3
    Operating System :: OS Independent
"""
classifiers = [s.strip() for s in classes.split('\n') if s]

description = 'Metabolomics Variational Autoencoders.'

this_dir = Path(__file__).parent
long_description = (this_dir / "README.md").read_text(encoding="utf-8")

# Single source of the version string
version_source = (this_dir / "metvae" / "__init__.py").read_text(encoding="utf-8")
version_match = re.search(r'^__version__\s*=\s*["\']([^"\']+)["\']', version_source, re.M)
if version_match is None:
    raise RuntimeError("Unable to find __version__ in metvae/__init__.py")
version = version_match.group(1)

setup(name='metvae',
      version=version,
      description=description,
      long_description=long_description,
      long_description_content_type="text/markdown",
      author="Huang Lin",
      author_email="huanglinfrederick@gmail.com",
      maintainer="Huang Lin",
      maintainer_email="huanglinfrederick@gmail.com",
      url="https://github.com/FrederickHuangLin/MetVAE-PyPI",
      license="MIT",
      license_files=["LICENSE"],
      packages=find_packages(exclude=['metvae.tests', 'metvae.tests.*', 'tests', 'tests.*']),
      install_requires=[
          'numpy>=1.21',
          'pandas>=1.5',
          'scipy>=1.7',
          'statsmodels>=0.13',
          'torch>=1.12',
          'networkx>=2.6',
          'tqdm>=4.62',
      ],
      extras_require={
          'logging': ['tensorboard>=2.9'],
          'test': ['pytest>=7'],
      },
      classifiers=classifiers,
      python_requires=">=3.9",
      entry_points={
          'console_scripts': [
              'metvae-cli=metvae.cli:main'
          ]
      }
      )
