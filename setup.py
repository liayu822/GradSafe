#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

setup(
    name="gradsafe",
    version="0.1.0",
    description="梯度特徵增強的LLM防禦系統",
    author="Liang Yu",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.22.0",
        "transformers>=4.25.0",
        "spacy>=3.5.0",
        "nltk>=3.8.0",
        "tqdm>=4.65.0",
        "scikit-learn>=1.2.0",
        "matplotlib>=3.5.0",
        "pandas>=1.5.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "pylint>=2.15.0",
        ],
    },
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)