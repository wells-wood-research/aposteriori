from setuptools import find_packages, setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="aposteriori",
    version="2.3.0",
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
    python_requires=">=3.8",
    packages=find_packages("src"),
    package_dir={"": "src"},
    setup_requires=[
        "Cython",
    ],
    entry_points={
        "console_scripts": [
            "make-frame-dataset=aposteriori.data_prep.cli:cli",
        ],
    },
    install_requires=[
        "ampal==1.5.1",
        "click==8.1.7",
        "h5py==3.8.0",
    ],
)
