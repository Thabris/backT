"""
Setup script for BackT - Professional Trading Backtesting Framework
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the contents of README file
this_directory = Path(__file__).parent
long_description = ""
readme_file = this_directory / "README.md"
if readme_file.exists():
    long_description = readme_file.read_text(encoding='utf-8')

# Read version from package
version = "0.1.0"

setup(
    name="backt",
    version=version,
    author="BackT Development Team",
    author_email="dev@backt.com",
    description="A professional trading backtesting framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/backt",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Financial and Insurance Industry",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Office/Business :: Financial :: Investment",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    python_requires=">=3.8",
    install_requires=[
        "pandas>=1.3.0",
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "yfinance>=0.1.70",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.9",
            "mypy>=0.910",
            "pre-commit>=2.15",
        ],
        "jupyter": [
            "jupyter>=1.0",
            "ipykernel>=6.0",
            "notebook>=6.4",
        ],
        "web": [
            "streamlit>=1.0",
            "plotly>=5.0",
        ],
        "advanced": [
            "numba>=0.56",
            "scikit-learn>=1.0",
            "statsmodels>=0.13",
        ],
        "data": [
            "quandl>=3.7",
            "alpha-vantage>=2.3",
            "pandas-datareader>=0.10",
        ]
    },
    entry_points={
        "console_scripts": [
            "backt=backt.api.cli:BacktestCLI.run_backtest",
        ],
    },
    include_package_data=True,
    package_data={
        "backt": ["*.json", "*.yaml", "*.yml"],
    },
    zip_safe=False,
    keywords=[
        "trading", "backtesting", "finance", "investment", "quantitative",
        "algorithmic trading", "strategy testing", "portfolio", "risk management"
    ],
    project_urls={
        "Bug Reports": "https://github.com/yourusername/backt/issues",
        "Source": "https://github.com/yourusername/backt",
        "Documentation": "https://backt.readthedocs.io",
    },
)