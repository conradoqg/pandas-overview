#!/usr/bin/env python

import sys
from setuptools import setup
from setuptools.command.test import test as TestCommand


class PyTest(TestCommand):
    def finalize_options(self):
        TestCommand.finalize_options(self)
        self.test_args = []
        self.test_suite = True

    def run_tests(self):
        import pytest
        errcode = pytest.main(self.test_args)
        sys.exit(errcode)


setup(name='pandas-overview',
      version='0.0.1',
      description='An extension to pandas describe function.',
      maintainer='Conrado Quilles Gomes',
      maintainer_email='conradoqg@gmail.com',
      url='https://github.com/conradoqg/pandas-overview',
      license='MIT',
      platforms='any',
      packages=['pandas_overview'],
      keywords=['pandas', 'data analysis', 'machine learning'],
      install_requires=[
          'numpy',
          'pandas',
      ],
      classifiers=[
          'Programming Language :: Python',
          'Operating System :: OS Independent',
          'Intended Audience :: Science/Research',
          'Topic :: Scientific/Engineering'
      ],
      tests_require=[
          'pytest',
          'xlrd'
      ],
      cmdclass={'test': PyTest})
