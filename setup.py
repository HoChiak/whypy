# -*- coding: utf-8 -*-

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="WhyPy",
    version="0.0.7",
    author="HoChiak",
    author_email="mark.henss@gmx.de",
    description="Toolbox for Causal Inference with Additive Noise Models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/HoChiak/whypy",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
