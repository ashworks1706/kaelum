from setuptools import setup, find_packages

setup(
    name="kaelum",
    version="1.0.0",
    description="Local reasoning models as cognitive middleware for commercial LLMs",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="KaelumAI Team",
    url="https://github.com/ashworks1706/KaelumAI",
    packages=find_packages(),
    install_requires=[
        "pydantic>=2.0.0",
        "httpx>=0.24.0",
        "sympy>=1.12",
        "python-dotenv>=1.0.0",
        "click>=8.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "ruff>=0.1.0",
            "mypy>=1.0.0",
        ],
        "rag": [
            "chromadb>=0.4.0",
            "qdrant-client>=1.6.0",
            "weaviate-client>=3.24.0",
        ],
        "langchain": [
            "langchain>=0.1.0",
            "langchain-community>=0.0.10",
        ],
        "all": [
            "chromadb>=0.4.0",
            "qdrant-client>=1.6.0",
            "weaviate-client>=3.24.0",
            "langchain>=0.1.0",
            "langchain-community>=0.0.10",
        ],
    },
    entry_points={
        'console_scripts': [
            'kaelum=kaelum.cli:cli',
        ],
    },
    python_requires=">=3.9",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
