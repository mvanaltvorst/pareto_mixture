from setuptools import setup

setup(
    name='pyparetomixture',
    version='0.1.0',
    description='This library fits a pareto mixture on data using L-BFGS to solve max likelihood.',
    author="Maurits van Altvorst",
    author_email="mvanaltvorst@icloud.com",
    packages=["pyparetomixture"],
    package_dir={"": "src"},
)