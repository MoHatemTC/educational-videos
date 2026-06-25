"""
Minimal setup.py — exists so that older pip versions (and `pip install -e .`)
work without needing PEP 660 / editable-install support in the build backend.
All real metadata lives in pyproject.toml.
"""

from setuptools import setup, find_packages

setup(
    name="vlm-render-mapper",
    version="1.0.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.11",
    install_requires=[
        "pydantic>=2.0",
        "Pillow>=10.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4",
            "pytest-cov>=4.1",
            "ruff>=0.1",
            "mypy>=1.5",
        ],
    },
    entry_points={
        "console_scripts": [
            "vlm-render-mapper=vlm_render_mapper.cli:main",
        ],
    },
)
