from setuptools import setup
from setuptools import find_packages

setup(name='processing_arithmetics',
      version='0.1',
      description='',
      url='https://github.com/dieuwkehupkes/processing_arithmetics',
      author='Sara Veldhoen, Dieuwke Hupkes',
      author_email='',
      install_require=['keras', 'matplotlib', 'sklearn', 'nltk', 'h5py'],
      packages=find_packages())
