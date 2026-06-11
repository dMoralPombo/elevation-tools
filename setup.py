from setuptools import setup, find_packages

setup(
    name="elevation-tools",
    version="0.1.0",
    description="Tools for ArcticDEM elevation profile and time series analysis",
    author="Diego Moral Pombo",
    author_email="d.moralpombo@lancaster.ac.uk",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "geopandas>=0.10.0",
        "rasterio>=1.2.0",
        "matplotlib>=3.5.0",
    ],
    python_requires=">=3.8",
)
EOF
