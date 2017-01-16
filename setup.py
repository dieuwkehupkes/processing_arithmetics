from setuptools import setup
from setuptools import find_packages

setup(name='processing_arithmetics',
      version='0.1',
      description='',
      url='https://github.com/dieuwkehupkes/processing_arithmetics',
      author='Sara Veldhoen, Dieuwke Hupkes',
      author_email='dieuwkehupkes@gmail.com',
      dependency_links=['https://github.com/dieuwkehupkes/keras'],
      install_requires=['matplotlib', 'sklearn', 'nltk', 'h5py'],
      packages=find_packages())
