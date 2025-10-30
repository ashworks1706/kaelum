"""Setup script for KaelumAI."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="kaelum",
    version="0.2.0",
    description="One line to make any LLM reason better",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Ash",
    author_email="ashworks1706@users.noreply.github.com",
    url="https://github.com/ashworks1706/KaelumAI",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.9",
    install_requires=[
        "fastapi>=0.100.0",
        "uvicorn[standard]>=0.23.0",
        "pydantic>=2.0.0",
        "openai>=1.0.0",
        "sympy>=1.12",
        "numpy>=1.24.0",
        "python-dotenv>=1.0.0",
        "httpx>=0.24.0",
    ],
    extras_require={
        "cache": ["redis>=5.0.0"],
        "dev": [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.0",
            "black>=23.0.0",
            "ruff>=0.1.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "kaelum=kaelum.cli:main",
        ],
    },
)
