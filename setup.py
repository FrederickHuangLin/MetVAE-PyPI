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

description = ('Metabolomics Variational Autoencoders.')


setup(name='metvae',
      version='0.0.2',
      description=description,
      long_description=open("README.md").read(),
      long_description_content_type="text/markdown",
      author_email="huanglinfrederick@gmail.com",
      maintainer_email="huanglinfrederick@gmail.com",
      packages=find_packages(exclude=['tests', 'tests.*']),  # Excludes tests
      install_requires=[
          'joblib>=1.2.0',
          'numpy>=1.20.0',
          'scipy',
          'statsmodels',
          'scikit-learn',
          'tensorboard',
          'tqdm',
          'pandas>=1.0.0',
          'torch>=1.8.0',
          'pytorch-lightning>=1.3.1'
      ],
      classifiers=classifiers,
      python_requires=">=3.9",
      entry_points = {
          'console_scripts': [
              'metvae-cli=metvae.cli:main'
          ]
      }
     )
