from setuptools import setup, find_namespace_packages

if __name__ == "__main__":
    setup(
        name="eps",
        description="Package used to compute EPS",
        package_dir={"": "src"},
        version=1.0,
        author="Coline Metta-Versmessen",
        packages=find_namespace_packages("./src"),
        python_requires=">=3.10",
        install_requires=[
            # "emoji>=1.6.1",
            "matplotlib>=3.3.1",
            "numpy>=1.19.1",
            # "pandas>=1.1.2",
            # "SQLAlchemy>=1.3.19",
            # "pytest>=6.2.5",
            # "tqdm>=4.0",
            # "GitPython>=3.1.13",
            # "dash>=2.3.1",
            # "dash_bootstrap_components>=1.1.0",
            # "frozendict>=2.2.1",
            # "plotly>=5.5.0",
            # "pycountry>=22.1.10",
            # "regex>=2022.1.18",
            # "SQLAlchemy>=1.4.31",
            # "regex>=2022.1.18",
            # "boto3>=v1.23.10",
            # "awswrangler>=2.12",
            # "pandas_flavor>=0.3",
            # "geopandas>=0.11",
            # "macro_q3>=1.0.0",
            # "pyyaml>=6.0",
            # "rich>=12.6.0",
            # "rapidfuzz>=2.13"
        ],
    )