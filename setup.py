from setuptools import setup, find_packages

setup(
    name="hypertunity",
    version="0.1",
    author="Georgi Dikov",
    author_email="gvdikov@gmail.com",
    url="https://github.com/gdikov/hypertunity",
    description="Framework for distributed black-box hyperparameter optimisation.",
    long_description=open("README.md").read(),
    packages=find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
    python_requires=">=3.6",
    install_requires=["dataclasses;python_version<'3.7'", "gpyopt==1.2.5", "matplotlib>=3.0"],
    extras_require={
        "tensorboard": ["tensorflow>=1.14.0", "tensorboard>=1.14.0"],
        "tests": ["pytest>=4.6.3"]
    },
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules"
    ]
)
