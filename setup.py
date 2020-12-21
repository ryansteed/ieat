from distutils.core import setup

setup(
    name='ieat',
    version='1.0',
    packages=['ieat',],
    license='Creative Commons Attribution-Noncommercial-Share Alike license',
    long_description=open('README.md').read(),
    install_requires=[
    	'transformers',
        'torch',
        'numpy',
        'matplotlib',
        'opencv-python',
        'tensorflow',
        'torchvision'
    ]
)