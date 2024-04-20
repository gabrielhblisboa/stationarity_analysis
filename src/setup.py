from setuptools import setup, find_packages

setup(
    name='noise_synthesis',
    version='0.1',
    packages=find_packages(),
    description='Noise Synthesis Library using the Kullback-Leibler Divergence',
    long_description=open('README.md').read(),
    author='Gabriel Lisboa, Fabio Oliveira',
    package_data={'noise_synthesis': ['data/*.wav']},
)
