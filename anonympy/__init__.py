from anonympy import pandas
from anonympy import images
from anonympy import sound
from anonympy import text
from anonympy import pdf


'''
Package for data anonymization of different data types
such as tabular, text, images and sound.
'''

__version__ = "0.3.7"
# Check if all dependencies have been installed
hard_dependencies = ("faker", "pandas", "numpy", "cv2", "pypdf",
                     "pytesseract", "pdf2image",
                     'transformers')
missing_dependencies = []

for dependency in hard_dependencies:
    try:
        __import__(dependency)
    except ImportError as e:
        missing_dependencies.append(f"{dependency}: {e}")

if missing_dependencies:
    raise ImportError(
        "Unable to import required dependencies:\n" +
        "\n".join(missing_dependencies)
    )
del hard_dependencies, dependency, missing_dependencies

__all__ = ["pandas", "images", "sound", "text", "pdf"]
