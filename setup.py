import re

from setuptools import setup, find_packages

with open("hypertunity/__init__.py", "r", encoding="utf8") as f:
    version = re.search(r"__version__ = [\'\"](.*?)[\'\"]", f.read()).group(1)

with open("README.md", "r", encoding="utf8") as f:
    readme = f.read()

required_packages = [
    "beautifultable>=0.7.0",
    "dataclasses;python_version<'3.7'",
    "gpy==1.9.8",
    "gpyopt==1.2.5",
    "joblib>=0.13.2",
    "matplotlib>=3.0",
    "numpy>=1.16",
    "tinydb>=3.13.0"
]

extras = {
    "tensorboard": ["tensorflow>=1.14.0", "tensorboard>=1.14.0"],
    "tests": ["pytest>=4.6.3", "pytest-timeout>=1.3.3"],
    "docs": ["sphinx>=2.2.0", "sphinx_rtd_theme>=0.4.3"]
}

classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Developers",
    "Intended Audience :: Education",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: Apache Software License",
    "Programming Language :: Python :: 3.6",
    "Programming Language :: Python :: 3.7",
    "Programming Language :: Python :: 3.8",
    "Topic :: Software Development :: Libraries",
    "Topic :: Software Development :: Libraries :: Python Modules"
]

setup(
    name="hypertunity",
    version=version,
    author="Georgi Dikov",
    author_email="gvdikov@gmail.com",
    url="https://github.com/gdikov/hypertunity",
    description="A toolset for distributed black-box hyperparameter optimisation.",
    long_description=readme,
    long_description_content_type='text/markdown',
    packages=find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
    python_requires=">=3.6",
    install_requires=required_packages,
    extras_require=extras,
    classifiers=classifiers
)
