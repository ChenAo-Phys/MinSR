from setuptools import setup, find_packages

setup(
    name='quantax',
    version='0.0.1',
    description='Flexible neural quantum states based on QuSpin, JAX, and Equinox',
    author='Ao Chen',
    author_email='chenao.phys@gmail.com',
    packages=find_packages(),
    python_requires=">=3.9,<3.10",
    install_requires=[
        'quspin>=0.3.7',
        'matplotlib>=3.8.0',
        'equinox>=0.11.4',
    ],
)