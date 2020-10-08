from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize, build_ext
import numpy
from os import path


this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

annlib_extension = Extension(
    name="annlib",
    sources=["pyann/annlib/annlib.pyx", "pyann/annlib/lib/ann2.cpp"],
    libraries=["NN"],
    library_dirs=["pyann/annlib/lib"],
    include_dirs=["pyann/annlib/lib", numpy.get_include(), "pyann/annlib/lib/ANNlib/include"],
    language= "c++"
)

setup(
    name="pyann",
    version = '0.0.1',
    description="Python wrapper for Arya and Mount's ANN library (v1.1.3)",
    long_description = long_description,
    long_description_content_type='text/markdown',
    author='Anna-Christina Nevison',
    author_email='annanev@umich.edu',
    setup_requires=[
        # Setuptools 18.0 properly handles Cython extensions.
        'setuptools>=18.0',
        'cython',
    ],
    install_requires=[
        'numpy',
        'pandas'
    ],
    packages = find_packages(),
    license='MIT',
    url='http://github.com/annacnev/pyann',
    ext_modules=cythonize([annlib_extension]),
    python_requires='>=3.6',
    test_suite='nose.collector',
    tests_require=['nose']
)

