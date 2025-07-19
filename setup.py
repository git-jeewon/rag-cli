"""Setup script for RAG CLI."""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

# Read requirements
requirements = []
with open("requirements.txt") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="rag-cli",
    version="0.1.0",
    author="RAG CLI Team",
    author_email="contact@example.com",
    description="A modular Python tool for ingesting, embedding, and querying unstructured data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/rag-cli",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "rag-cli=rag_cli.cli.main:app",
        ],
    },
    include_package_data=True,
    zip_safe=False,
) 