from pathlib import Path
from setuptools import find_packages, setup


with open("requirements.txt") as f:
    requirements = [
        line.strip()
        for line in f.readlines()
        if not line.startswith("-f")
    ]

with open("README.md", "r") as f:
    long_description = f.read()


setup(
    name="attentions",
    version="0.0.0",
    author="YeaMerci",
    author_email="entertomerci@gmail.com",
    description="""
    The package contains various types of attention mechanisms, 
    activation functions and other transformer building blocks 
    implemented as PyTorch layers for easy use ⭐️
    """,
    long_description=long_description,
    install_requires=requirements,
    long_description_content_type="text/markdown",
    url="https://github.com/YeaMerci/attentions.git",
    packages=find_packages(),
    include_package_data=True,
    keywords=[
        "torch", "PyTorch",
        "NLP", "Attention",
        "Self-attention",
        "Multi Head Attention",
        "Neural Network",
        "Transformers",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10"
)
