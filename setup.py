from setuptools import find_packages, setup

# Read the requirements from the requirements.txt file
with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name='fast_forward',
    version='0.5.2',
    packages=find_packages(),
    install_requires=required,
    author="Jurek Leonhardt (https://github.com/mrjleo), Bo van den Berg (https://github.com/bovdberg)",
)
