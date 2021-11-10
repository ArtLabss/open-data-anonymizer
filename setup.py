from setuptools import setup, find_packages

with open('README.md') as readme_file:
    readme = readme_file.read()

setup(
  author="ArtLabss",
  description="A Data Anonymization package for tabular, text, image and sound data",
  name="anonympy",
  version="0.1.0",
  packages=find_packages(include=["anonympy", "anonympy.*"]),
  install_requires=['pandas', 'faker', 'cape-python', 'numpy'],
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
  license="BSD license",
  long_description=readme,
  long_description_content_type='text/markdown',
  keywords='anonympy',    
  zip_safe=False,
)

# pip install -e .
# pip freeze > requirements.txt
