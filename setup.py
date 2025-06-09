from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name='csmooth',
    version='0.1.0',
    author='David G. Ellis',
    author_email='david.ellis@unmc.edu',
    description='A Python package for constrained smoothing of fMRI data',
    long_description_content_type='text/markdown',
    url='https://github.com/ellisdg/ConstrainedSmoothing',
    packages=find_packages(),
    install_requires=requirements,
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',  # Choose an appropriate license
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.11',
)

