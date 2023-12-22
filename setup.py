import sys

import setuptools

# READ README.md for long description on PyPi.
try:
    long_description = open("README.md", encoding="utf-8").read()
except Exception as e:
    sys.stderr.write(f"Failed to read README.md:\n  {e}\n")
    sys.stderr.flush()
    long_description = ""

setuptools.setup(
    name="independent_component_analysis",
    author="Nour Bouayed, Yvann Le Fay, Zineb Bentires",
    description="Non linear independent component analysis and variational autoencoder for approximating the joint distribution over some observed and latent variables",
    long_description=long_description,
    version="0.1",
    packages=setuptools.find_packages(),
    install_requires=[
        "jax>=0.3.25",
        "jaxlib>=0.3.25",
        "pytest",
        "numpy>=1.24.3",
        "flax",
        "optax",
    ],
    long_description_content_type="text/markdown",
    keywords="probabilistic graphical models generative models component analysis variational autoencoder nonlinear model",
    license="MIT",
    license_files=("LICENSE",),
)
