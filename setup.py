from setuptools import setup, find_packages
from src.aposteriori.config import MAKE_FRAME_DATASET_VER

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="aposteriori",
    version=MAKE_FRAME_DATASET_VER,
    author="Wells Wood Research Group",
    author_email="chris.wood@ed.ac.uk",
    description="A library for the voxelization of protein structures for protein design.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/wells-wood-research/aposteriori",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    packages=find_packages("src"),
    package_dir={"": "src"},
)
