"""
Setup script for Vanna.AI custom implementation.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

# Read requirements
requirements_path = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_path.exists():
    with open(requirements_path, 'r', encoding='utf-8') as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="vanna-ai-custom",
    version="1.0.0",
    author="Vanna.AI Custom Implementation",
    author_email="support@vanna.ai",
    description="Professional text-to-SQL generation using RAG and LLM",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-org/vanna-custom",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Database",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.5.0",
        ],
        "api": [
            "fastapi>=0.100.0",
            "uvicorn>=0.23.0",
        ],
        "frontend": [
            "streamlit>=1.28.0",
        ],
        "all": [
            "fastapi>=0.100.0",
            "uvicorn>=0.23.0",
            "streamlit>=1.28.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "vanna-api=api.main:main",
            "vanna-ui=frontend.streamlit_app:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.md", "*.txt", "*.yml", "*.yaml", "*.json"],
    },
    zip_safe=False,
) 