
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()


setuptools.setup(
    name="cost2fitness", 
    version="1.0.1",
    author="Demetry Pascal",
    author_email="qtckpuhdsa@gmail.com",
    maintainer = ['Demetry Pascal'],
    description="PyPI package for conversion cost values (less is better) to fitness values (more is better) and vice versa",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/PasaOpasen/cost2fitness",
    keywords=[ 'optimization', 'evolutionary algorithms', 'fast', 'easy', 'evolution', 'generator', 'simple', 'converter', 'min2max', 'max2min', 'barplots'],
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    install_requires=['numpy', 'matplotlib']
    
    )





