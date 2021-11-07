from setuptools import setup, find_packages

setup(
  author="ArtLabss",
  description="A Data Anonymization package for tabular, text, image and sound data",
  name="anonympy",
  version="0.1.0",
  packages=find_packages(include=["anonympy", "anonympy.*"]),
)

# pip install -e .
