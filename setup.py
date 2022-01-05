import re
from setuptools import setup, find_packages


def readme():
    """Return the contents of the project README file."""
    with open('README.md') as f:
        return f.read()


version = re.search(r"__version__ = ['\"]([^'\"]*)['\"]", open('atiim/__init__.py').read(), re.M).group(1)

setup(
    name='atiim',
    version=version,
    packages=find_packages(),
    url='https://github.com/crvernon/atiim',
    license='BSD-2-Clause',
    author='André M. Coleman; Chris R. Vernon',
    description='The Area Time Inundation Index Model',
    long_description=readme(),
    long_description_content_type="text/markdown",
    python_requires='>=3.7.*, <4',
    include_package_data=True,
    install_requires=[
        'numpy>=1.19.4',
        'pandas>=1.1.4',
        'rasterio>=1.2.3',
        'requests>=2.25.1',
        'joblib>=1.0.1',
        'matplotlib>=3.3.3',
        'seaborn>=0.11.1',
        'fiona>=1.8.19',
        'pyproj>=3.0.1',
        'rtree>=0.9.7',
        'shapely>=1.7.1',
        'geopandas>=0.9.0'
    ],
    extras_require={
        'dev': [
            'build>=0.5.1',
            'nbsphinx>=0.8.6',
            'setuptools>=57.0.0',
            'sphinx>=4.0.2',
            'sphinx-panels>=0.6.0',
            'sphinx-rtd-theme>=0.5.2',
            'twine>=3.4.1',
            'pytest>=6.2.4',
            'pytest-cov>=2.12.1'
        ]
    }
)
