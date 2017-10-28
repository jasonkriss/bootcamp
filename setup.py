from setuptools import setup, find_packages

VERSION = '0.1.0'

setup(
    name='bootcamp',
    version=VERSION,
    author='Jason Kriss',
    author_email='jasonkriss@gmail.com',
    url='https://github.com/jasonkriss/bootcamp',
    description='Reinforcement learning with PyTorch and OpenAI Gym',
    license='MIT',
    packages=find_packages(),
    zip_safe=True,
    install_requires=[
        'numpy',
        'torch',
        'gym'
    ]
)
