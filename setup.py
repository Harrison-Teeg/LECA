import codecs
from setuptools import find_packages, setup


DISTNAME = "LECA"
VERSION = "0.0.1"
DESCRIPTION = (
    "A python library for modeling" 
    "liquid electrolyte behavior."
)
with codecs.open("README.rst", encoding="utf-8-sig") as f:
    LONG_DESCRIPTION = f.read()
LONG_DESCRIPTION_CONTENT_TYPE = "text/x-rst"
LICENSE = "MIT"
MAINTAINER = "Harrison Martin"
MAINTAINER_EMAIL = (
    "h_mart05@uni-muenster.de"
)
PYTHON_REQUIRES = ">=3.8"
PACKAGES = find_packages()
SETUP_REQUIRES = [
        "Cython==0.29.36"
    ]
INSTALL_REQUIRES = [
        "numpy>=1.22.3",
        "pandas>=1.4.2",
        "scipy>=1.8.1",
        "uncertainties>=3.1.7",
        "seaborn>=0.11.2",
        "matplotlib>=3.5.1",
        "scikit-learn>=1.3.1",
        "hdbscan>=0.8.28",
        "GPyOpt>=1.2.6",
        "mapie==0.6.5",
        "packaging"
    ]
EXTRAS_REQUIRE = {
    "tests": [],
    "docs": [
        "matplotlib",
        "numpydoc",
        "pandas",
        "sphinx",
        "nbsphinx",
        "sphinx-gallery",
        "sphinx_rtd_theme",
        "typing_extensions"
    ]
}
CLASSIFIERS = [
    "Intended Audience :: Science/Research",
    "Intended Audience :: Developers",
    "License :: MIT",
    "Topic :: Software Development",
    "Topic :: Scientific/Engineering",
    "Operating System :: Microsoft :: Windows",
    "Operating System :: POSIX",
    "Operating System :: Unix",
    "Operating System :: MacOS",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10"
]

setup(
    name=DISTNAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type=LONG_DESCRIPTION_CONTENT_TYPE,
    license=LICENSE,
    maintainer=MAINTAINER,
    maintainer_email=MAINTAINER_EMAIL,
    packages=PACKAGES,
    python_requires=PYTHON_REQUIRES,
    setup_requires=SETUP_REQUIRES,
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
    classifiers=CLASSIFIERS,
    zip_safe=False  # the package can run out of an .egg file
)
