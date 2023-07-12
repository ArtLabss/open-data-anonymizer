import os
from setuptools import setup, find_packages


def package_files(directory):
    paths = []
    for (path, directories, filenames) in os.walk(directory):
        for filename in filenames:
            paths.append(os.path.join('..', path, filename))
    return paths


with open('README.md', encoding='utf8') as readme_file:
    readme = readme_file.read()

extra_files = package_files('anonympy')

setup(
  author="ArtLabs",
  author_email="art@artlabs.tech",
  description="A Data Anonymization package for tabular, image and PDF data",
  name="anonympy",
  version="0.3.7",

  packages=find_packages(exclude=['tests*']),
  package_data={'anonympy': extra_files},
  install_requires=['faker', 'scikit-learn', 'opencv_python',
                    'texttable', 'setuptools', 'numpy', 'pandas', 'validators',
                    'pycryptodome', 'requests', 'pyyaml', 'rfc3339',
                    'pytesseract', 'pypdf', 'poppler-utils', 'pdf2image',
                    'transformers', 'cape-dataframes>=0.3.1'],

  python_requires='>=3.6',
  url='https://github.com/ArtLabss/open-data-anonimizer',
  classifiers=[
    'Development Status :: 3 - Alpha',
    'Intended Audience :: Developers',
    'License :: OSI Approved :: BSD License',
    'Natural Language :: English',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
    'Programming Language :: Python :: 3.9',
    'Programming Language :: Python :: 3.10',
    'Operating System :: OS Independent',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'Topic :: Security :: Cryptography',
    ],
  license="BSD license",
  long_description=readme,
  long_description_content_type='text/markdown',
  keywords='anonympy',
  zip_safe=False,
)
