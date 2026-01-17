"""Setup script for SeraphynAI package."""

from setuptools import setup, find_packages
import os

# Read version from __version__.py
version = {}
with open("seraphynai/__version__.py") as f:
    exec(f.read(), version)

# Read README for long description
with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

# Core dependencies
install_requires = [
    "numpy>=1.24.0",
    "scipy>=1.10.0",
]

# API dependencies
api_requires = [
    "fastapi>=0.104.0",
    "uvicorn[standard]>=0.24.0",
    "pydantic>=2.0.0",
    "python-multipart>=0.0.6",
    "websockets>=12.0",
    "python-jose[cryptography]>=3.3.0",
    "passlib[bcrypt]>=1.7.4",
    "python-dotenv>=1.0.0",
    "slowapi>=0.1.9",
]

# Storage dependencies
storage_requires = [
    "sqlalchemy>=2.0.0",
    "alembic>=1.12.0",
    "psycopg2-binary>=2.9.9",
]

# Monitoring dependencies
monitoring_requires = [
    "prometheus-client>=0.19.0",
    "structlog>=23.2.0",
]

# Development dependencies
dev_requires = [
    "pytest>=7.4.0",
    "pytest-cov>=4.1.0",
    "pytest-asyncio>=0.21.0",
    "black>=23.11.0",
    "flake8>=6.1.0",
    "mypy>=1.7.0",
    "isort>=5.12.0",
    "pylint>=3.0.0",
]

# Documentation dependencies
docs_requires = [
    "mkdocs>=1.5.0",
    "mkdocs-material>=9.4.0",
    "mkdocstrings[python]>=0.24.0",
]

# Notebook dependencies
notebook_requires = [
    "jupyter>=1.0.0",
    "matplotlib>=3.8.0",
    "seaborn>=0.13.0",
]

# All dependencies
all_requires = (
    install_requires +
    api_requires +
    storage_requires +
    monitoring_requires +
    dev_requires +
    docs_requires +
    notebook_requires
)

setup(
    name="seraphynai",
    version=version["__version__"],
    author=version["__author__"],
    description=version["__description__"],
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ManuelMello-dev/seraphynai",
    packages=find_packages(exclude=["tests", "tests.*", "docs", "notebooks"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.9",
    install_requires=install_requires,
    extras_require={
        "api": api_requires,
        "storage": storage_requires,
        "monitoring": monitoring_requires,
        "dev": dev_requires,
        "docs": docs_requires,
        "notebook": notebook_requires,
        "all": all_requires,
    },
    entry_points={
        "console_scripts": [
            "seraphynai=seraphynai.cli:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
