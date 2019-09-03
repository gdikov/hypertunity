from setuptools import setup, find_packages

required_packages = [
    "dataclasses;python_version<'3.7'",
    "numpy>=1.16",
    "gpyopt==1.2.5",
    "matplotlib>=3.0",
    "joblib>=0.13.2",
    "beautifultable>=0.7.0"
]

extras = {
    "tensorboard": ["tensorflow>=1.14.0", "tensorboard>=1.14.0"],
    "tests": ["pytest>=4.6.3", "pytest-timeout>=1.3.3"]
}

classifiers = [
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

setup(
    name="hypertunity",
    version="0.3dev2",
    author="Georgi Dikov",
    author_email="gvdikov@gmail.com",
    url="https://github.com/gdikov/hypertunity",
    description="A toolset for distributed black-box hyperparameter optimisation.",
    long_description=open("README.md").read(),
    packages=find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
    python_requires=">=3.6",
    install_requires=required_packages,
    extras_require=extras,
    classifiers=classifiers
)
