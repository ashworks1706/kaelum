from setuptools import setup, find_packages

setup(
    name="kaelum",
    version="0.1.0",
    description="Reasoning Acceleration Layer for Lightweight LLMs",
    author="KaelumAI Team",
    packages=find_packages(),
    install_requires=[
        "openai>=1.0.0",
        "pydantic>=2.0.0",
        "sympy>=1.12",
    ],
    extras_require={
        "dev": [
            "jupyter",
            "pytest",
        ],
        "rag": [
            "chromadb",
            "pinecone-client",
            "weaviate-client",
        ],
    },
    python_requires=">=3.9",
)
