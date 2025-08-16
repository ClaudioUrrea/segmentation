# setup.py
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="intelligent-mining-systems",
    version="1.0.0",
    author="Claudio Urrea, Maximiliano Vélez",
    author_email="claudio.urrea@usach.cl",
    description="Intelligent Systems for Autonomous Mining Operations with Single-Domain Generalization",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ClaudioUrrea/segmentation",
    project_urls={
        "Paper": "https://doi.org/10.3390/xxxxx",
        "Dataset": "https://doi.org/10.6084/m9.figshare.29897300",
        "Bug Tracker": "https://github.com/ClaudioUrrea/segmentation/issues",
    },
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=21.0",
            "flake8>=3.9",
            "isort>=5.9",
            "mypy>=0.910",
        ],
        "docs": [
            "sphinx>=4.0",
            "sphinx-rtd-theme>=0.5",
            "myst-parser>=0.15",
        ],
    },
    entry_points={
        "console_scripts": [
            "mining-inference=utils.inference:main",
            "mining-train=utils.training:main",
            "mining-eval=utils.evaluation:main",
        ],
    },
    include_package_data=True,
    package_data={
        "intelligent_mining": [
            "configs/*.py",
            "data/*.csv",
            "docs/*.md",
        ],
    },
    zip_safe=False,
)
