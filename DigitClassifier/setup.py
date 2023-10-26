from setuptools import setup, find_packages

setup(
    author="daven-c",
    description="draw and classify a number from 0-9",
    packages=find_packages(),
    requires=["keras", "numpy", "pandas", "matplotlib",
              "seaborn", "typing", "tensorflow", "pygame"],
)
