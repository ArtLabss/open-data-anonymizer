from setuptools import setup, find_packages

setup(
  author="ArtLabss",
  description="A Data Anonymization package for tabular, text, image and sound data",
  name="anonympy",
  version="0.1.0",
  packages=find_packages(include=["anonympy", "anonympy.*"]),
  install_requires=['pandas>=1.0', 'scipy=1.1', 'matplotlib>=2.2.1,<3'],
  python_requires='>=3.6.*',
  classifiers=[
    'Development Status :: 2 - Pre-Alpha',
    'Intended Audience :: Developers',
    'License :: OSI Approved :: BSD License',
    'Natural Language :: English',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
    ],
)

# pip install -e .
# pip freeze > requirements.txt
