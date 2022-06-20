import os
import pathlib
from typing import Dict, List, Tuple, Union

import pytesseract
import requests
from PIL import Image
from pdf2image import convert_from_bytes, convert_from_path
from transformers import pipeline

from anonympy.pdf.utils_pdf import alter_metadata, \
     draw_black_box_pytesseract, find_EOI, \
     find_coordinates_pytesseract, find_emails, \
     find_months, find_numbers


class pdfAnonymizer(object):
    """
    Initializes pdfAnonymizer object

    Parameters:
    ----------
    path_to_pdf: Union[str, None], default None
        Any valid string path that points to PDF file.

    url: Union[str, None], default None
        Any valid URL, where the PDF will be fetched from.

    pytesseract_path: Union[str, pathlib.Path, None], default None
        `None` if Tesseract path is added to system environment.
        Otherwise, pass a valid string path to the binary.

    poppler_path: Union[str, pathlib.Path, None], default None
        `None` if Poppler path is specified in environment variable.
        Otherwise, pass poppler path.

    model: str, default "dbmdz/bert-large-cased-finetuned-conll03-english"
        The model that will be used by `transformers.pipeline`.
        If not specified, default model for NER will be used.

    tokenizer: str, default "dbmdz/bert-large-cased-finetuned-conll03-english"
        The tokenizer that will be used by `transformers.pipeline`.
        If not specified, default one will be used.

    Returns:
    ----------
    `pdfAnonymizer` object

    Raises
    ----------
    Exception:

        * If `path_to_pdf` doesn't exist

        * If file specified at `path_to_pdf` is not PDF

        * If `url` can't be fetched

        * If neither `path_to_pdf` nor `url` aren't specified

        * If both `path_to_pdf` and `url` are specified

    Examples
    ----------
    >>> from anonympy.pdf import pdfAnonymizer

    Pytesseract and poppler paths aren't specified in our environmental
    variables. Therefore, we'll have to specify them

    >>> anonym = pdfAnonymizer(path_to_pdf = 'Downloads\\test.pdf',
                               pytesseract_path = "C:\\Program Files
                               \\Tesseract-OCR\\tesseract.exe",
                               poppler_path = "C:\\Users\\shakhansho\\Downloads
                               \\Release-22.01.0-0\\poppler-22.01.0\\Library
                               \\bin",
                               model = "dslim/bert-base-NER",
                               tokenizer="dslim/bert-base-NER")
    """

    def __init__(self,
                 path_to_pdf: Union[str, None] = None,
                 url: Union[str, None] = None,
                 # path_to_folder: str = None,
                 pytesseract_path: Union[str, pathlib.Path, None] = None,
                 poppler_path: Union[str, pathlib.Path, None] = None,
                 model: str = "dbmdz/bert-large-cased-"
                              "finetuned-conll03-english",
                 tokenizer: str = "dbmdz/bert-large-cased"
                                  "-finetuned-conll03-english"):

        if path_to_pdf is not None:
            if not os.path.exists(path_to_pdf):
                raise Exception(f"Can't find PDF file at `{path_to_pdf}`")
            elif not path_to_pdf.endswith('.pdf'):
                raise Exception("`String path should end with `.pdf`. "
                                f"But {path_to_pdf[-4:]} was found.")

        if (path_to_pdf is None) and (url is None):
            raise Exception('Neither `path_to_pdf` nor `url` are specified!'
                            ' Please provide an input PDF file')
        elif (path_to_pdf is not None) and (url is not None):
            raise Exception('Please provide either `path_to_pdf` or `url`')

        # elif path_to_folder is not None:
        #     if not os.path.isdir(path_to_folder):
        #         raise Exception(f"Can't find folder at `{path_to_folder}`")
        #     else:
        #         for file in os.listdir(path_to_folder):
        #             if not file.endswith('.pdf'):
        #                 raise Exception(f"`Folder should contain only `.pdf`"
        #                     " files. But {file[-4:]} was found.")

        if pytesseract_path is not None:
            pytesseract.pytesseract.tesseract_cmd = pytesseract_path

        if poppler_path is not None:
            self._poppler_path = poppler_path
        else:
            self._poppler_path = None

        self._nlp = pipeline("ner",
                             aggregation_strategy="simple",
                             model=model,
                             tokenizer=tokenizer)
        self.path_to_pdf = path_to_pdf
        self.url = url
        # self.path_to_folder = path_to_folder
        self.number_of_pages = None  # changes in `pdfAnonymizer.pdf2images`
        self.images = []  # anonymized images that will be converted to PDF
        self.texts = []  # extracted text from images
        self.bbox = []  # bounding boxes of PII_objects
        self.pages_data = []  # data that will be returned by `images2text`
        self.PII_objects = []  # Personal Identifiable Information

    def anonymize(self,
                  output_path: str,
                  remove_metadata: bool = True,
                  fill: str = 'black',
                  outline: str = 'black') -> None:
        """
        Function for automatic PDF anonymization.
        PDF to Image -> Image to Text -> NER on Text -> Black Box ->
        Image to PDF -> Remove MetaData

        Parameters:
        ----------
        output_path: str
            String path together with the output file name that ends with .pdf

        remove_metadata: bool, default True
            If True PDF file's metadata will be replaced with
            {'Author': 'Unknown', 'Title': 'Title'}.

        fill: str, default 'black'
            The color to fill the boxes when covering sensetive information.

        outline: str, default 'black'
            Border color of the boxes.

        Returns:
        ----------
        None
            A new PDF file will be created with the name and at locaiton
            that was passed to `output_path`

        Raises
        ----------
        Exception
            * If `output_path` doesn't end with `.pdf`

        Examples
        ----------
        >>> from anonympy.pdf import pdfAnonymizer

        Initializing `pdfAnonymizer` object with specifying paths to binaries

        >>> anonym = pdfAnonymizer(path_to_pdf = "Downloads\\test.pdf",
                                   pytesseract_path = "C:\\Program Files\\
                                   Tesseract-OCR\\tesseract.exe",
                                   poppler_path = "C:\\Users\\shakhansho\\
                                   Downloads\\Release-22.01.0-0\\
                                   poppler-22.01.0\\Library\\bin")

        Calling `anonymize` function to create `output.pdf` in current
        directory and red covering boxes

        >>> anonym.anonymize(output_path = 'output.pdf',
                             remove_metadata = True,
                             fill = 'red',
                             outline = 'red')
        """
        if not output_path.endswith('.pdf'):
            raise Exception("`String path should end with `.pdf`."
                            f" But {output_path[-4:]} was found.")

        self.pdf2images()
        page_number = 1

        if self.number_of_pages == 1:
            # return str if 1 page PDF, else a list of str
            self.images2text(self.images)
            ner = self._nlp(self.texts[0])

            find_emails(text=self.texts[0], matches=self.PII_objects)
            find_numbers(text=self.texts[0], matches=self.PII_objects)
            find_months(text=self.texts[0], matches=self.PII_objects)

            find_EOI(pipeline=ner, matches=self.PII_objects, EOI="PER")
            find_EOI(pipeline=ner, matches=self.PII_objects, EOI="ORG")
            find_EOI(pipeline=ner, matches=self.PII_objects, EOI="LOC")

            find_coordinates_pytesseract(matches=self.PII_objects,
                                         data=self.pages_data[page_number - 1],
                                         bbox=self.bbox)

            self.cover_box(image=self.images[0],
                           bbox=self.bbox,
                           fill=fill,
                           outline=outline)
        else:
            self.images2text(self.images)

            for excerpt in self.texts:
                temp_pii = []
                temp_bbox = []
                ner = self._nlp(excerpt)

                find_emails(text=excerpt, matches=temp_pii)
                find_numbers(text=excerpt, matches=temp_pii)
                find_months(text=excerpt, matches=temp_pii)

                find_EOI(pipeline=ner, matches=temp_pii, EOI="PER")
                find_EOI(pipeline=ner, matches=temp_pii, EOI="ORG")
                find_EOI(pipeline=ner, matches=temp_pii, EOI="LOC")

                find_coordinates_pytesseract(  # noqa: F405
                    matches=temp_pii,
                    data=self.pages_data[page_number - 1],
                    bbox=temp_bbox)
                self.cover_box(self.images[page_number - 1],
                               temp_bbox, fill=fill,
                               outline=outline)

                self.PII_objects.append({f'page_{page_number}': temp_pii})
                self.bbox.append({f'page_{page_number}': temp_bbox})

                page_number += 1

        path = os.path.split(output_path)
        temp = os.path.join(path[0], 'temp.pdf')

        if self.number_of_pages == 1:
            if remove_metadata:
                self.images[0].save(temp)
                alter_metadata(temp, output_path)
            else:
                self.images[0].save(output_path)

        else:
            img1 = self.images.pop(0)
            if remove_metadata:
                img1.save(temp, save_all=True, append_images=self.images)
                alter_metadata(temp, output_path)
            else:
                img1.save(output_path,
                          save_all=True,
                          append_images=self.images)

    def pdf2images(self) -> None:
        """
        Convert PDF file to a list of images.

        Wrapper for `convert_from_path` and `convert_from_binary` functions
        from `pdf2image` library.

        Returns:
        ----------
        None
            A list of PIL images will be stored in `images` attribute

        Notes:
        ----------
        Function uses PDF file that was passed to `path_to_pdf`
        when initializing the class.
        `poppler_path` will also be required here, unless it's already in
        system variables.

        Examples
        ----------
        >>> from anonympy.pdf import pdfAnonymizer
        >>> anonym = pdfAnonymizer(path_to_pdf = "Downloads\\test.pdf",
                                   pytesseract_path = "C:\\Program Files\\
                                   Tesseract-OCR\\tesseract.exe",
                                   poppler_path = "C:\\Users\\shakhansho\\
                                   Downloads\\Release-22.01.0-0\\
                                   poppler-22.01.0\\Library\\bin")

        Storing resulting PIL images in a variable

        >>> anonym.pdf2images()
        >>> print(anonym.number_of_pages)
        ... 1
        >>> display(anonym.images[0])  # display first page
        """
        if self._poppler_path is None:
            if self.url is None:
                self.images = convert_from_path(self.path_to_pdf)
            else:
                pdf = requests.get(self.url)
                self.images = convert_from_bytes(pdf.content)

        elif self.url is None:
            self.images = convert_from_path(
                self.path_to_pdf,
                poppler_path=self._poppler_path)

        else:
            pdf = requests.get(self.url)
            self.images = convert_from_bytes(
                pdf.content,
                poppler_path=self._poppler_path)

        self.number_of_pages = len(self.images)

    def images2text(self, images: List[Image.Image]) -> None:
        """
        Extract text from an image.

        Wrapper for `image_to_data` function from `pytesseract` library.

        Parameters:
        ----------
        images: List[Image.Image]
            A list of PIL Images

        Returns:
        ----------
        None
            List of strings will be stored in `images` attribute. Where,
            string is a text extracted from the image.
            Index of each string represents the page number.

        Notes:
        ----------
        After running the function, output returned by
        `pytesseract.image_to_data`
        will be stored in `pdfAnonymizer.pages_data` attribute

        Examples
        ----------
        >>> from anonympy.pdf import pdfAnonymizer
        >>> anonym = pdfAnonymizer(path_to_pdf = "Downloads\\test.pdf",
                                   pytesseract_path = "C:\\Program Files\\
                                   Tesseract-OCR\\tesseract.exe",
                                   poppler_path = "C:\\Users\\shakhansho\\
                                   Downloads\\Release-22.01.0-0\\
                                   poppler-22.01.0\\Library\\bin")
        >>> anonym.pdf2images()

        Passing a list of PIL images

        >>> anonym.images2text(anonym.images)
        >>> print(anonym.texts[0])  # text extracted from first page
        """
        for image in images:
            page = pytesseract.image_to_data(image, output_type="dict")
            self.pages_data.append(page)
            excerpt = " ".join((line.strip() for line in page["text"] if line))
            self.texts.append(excerpt)

    def find_emails(self, text: List[str]) -> Dict[str, List[Tuple[int]]]:
        """
        Find emails within the string and return the coordinates.

        Wrapper for `find_emails` function from `anonympy.pdf.utils` module.

        Parameters:
        ----------
        text: List[str]
            A list of strings to search for emails.

        Returns:
        ----------
        Dict[str, List[Tuple[int, int, int, int]]]
            A dictionary is returned with page numbers as a key and a tuple
            storing 4 integer coordinates as a value.

        Notes
        ----------
        Underlying function uses RegEx to find emails.

        Examples
        ----------
        >>> from anonympy.pdf import pdfAnonymizer
        >>> anonym = pdfAnonymizer(path_to_pdf = "Downloads\\test.pdf",
                                   pytesseract_path = "C:\\Program Files\\
                                   Tesseract-OCR\\tesseract.exe",
                                   poppler_path = "C:\\Users\\shakhansho\\
                                   Downloads\\Release-22.01.0-0\\
                                   poppler-22.01.0\\Library\\bin")
        >>> images = anonym.pdf2images()
        >>> texts = anonym.images2text(images)

        Passing a list of strings to the function

        >>> emails = anonym.find_emails(texts)
        """
        coords = {}
        for page_number, excerpt in enumerate(text, 1):
            emails = []
            bbox = []

            find_emails(excerpt, emails)
            find_coordinates_pytesseract(emails,
                                         self.pages_data[page_number - 1],
                                         bbox)
            coords[f'page_{page_number}'] = bbox
        return coords

    def find_numbers(self, text: List[str]) -> Dict[str, List[Tuple[int]]]:
        """
        Find numbers within the string and return the coordinates.

        Wrapper for `find_numbers` function from `anonympy.pdf.utils` module.

        Parameters:
        ----------
        text: List[str]
            A list of strings to search for numbers.

        Returns:
        ----------
        Dict[str, List[Tuple[int, int, int, int]]]
            A dictionary is returned with page numbers as a key and a tuple
            storing 4 integer coordinates as a value.

        Notes
        ----------
        Underlying function uses RegEx to find numbers.

        Examples
        ----------
        >>> from anonympy.pdf import pdfAnonymizer
        >>> anonym = pdfAnonymizer(path_to_pdf = "Downloads\\test.pdf",
                                   pytesseract_path = "C:\\Program Files\\
                                   Tesseract-OCR\\tesseract.exe",
                                   poppler_path = "C:\\Users\\shakhansho\\
                                   Downloads\\Release-22.01.0-0\\
                                   poppler-22.01.0\\Library\\bin")
        >>> images = anonym.pdf2images()
        >>> text = anonym.images2text(images)

        Passing a list of strings to the function

        >>> numbers = anonym.find_numbers(text)
        """
        coords = {}
        for page_number, excerpt in enumerate(text, 1):
            bbox = []
            numbers = []

            find_numbers(excerpt, numbers)
            find_coordinates_pytesseract(numbers,
                                         self.pages_data[page_number - 1],
                                         bbox)
            coords[f'page_{page_number}'] = bbox
        return coords

    def find_months(self, text: List[str]) -> Dict[str, List[Tuple[int]]]:
        """
        Find months within the string and return the coordinates.

        Wrapper for `find_months` function from `anonympy.pdf.utils` module.

        Parameters:
        ----------
        text: List[str]
            A list of strings to search for months.

        Returns:
        ----------
        Dict[str, List[Tuple[int, int, int, int]]]
            A dictionary is returned with page numbers as a key and a tuple
            storing 4 integer coordinates as a value.

        Notes
        ----------
        Underlying function uses RegEx to find months.

        Examples
        ----------
        >>> from anonympy.pdf import pdfAnonymizer
        >>> anonym = pdfAnonymizer(path_to_pdf = "Downloads\\test.pdf",
                                   pytesseract_path = "C:\\Program Files\\
                                   Tesseract-OCR\\tesseract.exe",
                                   poppler_path = "C:\\Users\\shakhansho\\
                                   Downloads\\Release-22.01.0-0\\
                                   poppler-22.01.0\\Library\\bin")
        >>> images = anonym.pdf2images()
        >>> text = anonym.images2text(images)

        Passing a list of strings to the function

        >>> numbers = anonym.find_months(text)
        """
        coords = {}
        for page_number, excerpt in enumerate(text, 1):
            months = []
            bbox = []

            find_months(excerpt, months)
            find_coordinates_pytesseract(months,
                                         self.pages_data[page_number - 1],
                                         bbox)
            coords[f'page_{page_number}'] = bbox
        return coords

    def _find_EOI(self,
                  text: List[str],
                  EOI: str) -> Dict[str, List[Tuple[int]]]:
        """
        Generic function to find Entity of Interest (EOI) within the string and
        return the coordinates.

        Wrapper for `find_EOI` function from `anonympy.pdf.utils` module.

        Parameters:
        ----------
        text: List[str]
            A list of strings to search for EOI.
        EOI: str
            Accepted values are "PER" - people names, "ORG" - ogranization
            names/titles, "LOC" - locations.

        Returns:
        ----------
        Dict[str, List[Tuple[int, int, int, int]]]
            A dictionary is returned with page numbers as a key and a tuple
            storing 4 integer coordinates as a value.

        Examples
        ----------
        >>> from anonympy.pdf import pdfAnonymizer
        >>> anonym = pdfAnonymizer(path_to_pdf = "Downloads\\test.pdf",
                                   pytesseract_path = "C:\\Program Files\\
                                   Tesseract-OCR\\tesseract.exe",
                                   poppler_path = "C:\\Users\\shakhansho\\
                                   Downloads\\Release-22.01.0-0\\
                                   poppler-22.01.0\\Library\\bin")
        >>> images = anonym.pdf2images()
        >>> text = anonym.images2text(images)

        Passing a list of strings to the function and specifying EOI

        >>> names = anonym._find_EOI(text, "PER")
        """
        coords = {}
        for page_number, excerpt in enumerate(text, 1):
            ner = self._nlp(excerpt)
            names = []
            bbox = []

            find_EOI(ner, names, EOI)
            find_coordinates_pytesseract(names,
                                         self.pages_data[page_number - 1],
                                         bbox)
            coords[f'page_{page_number}'] = bbox
        return coords

    def find_ORG(self, text: List[str]) -> Dict[str, List[Tuple[int]]]:
        """
        Find organizations' names within the string and return the coordinates.

        Wrapper for `_find_EOI` function with the argument of `EOI = "ORG"`.

        Parameters:
        ----------
        text: List[str]
            A list of strings to search for EOI.

        Returns:
        ----------
        Dict[str, List[Tuple[int, int, int, int]]]
            A dictionary is returned with page numbers as a key and a tuple
            storing 4 integer coordinates as a value.

        Examples
        ----------
        >>> from anonympy.pdf import pdfAnonymizer
        >>> anonym = pdfAnonymizer(path_to_pdf = "Downloads\\test.pdf",
                                   pytesseract_path = "C:\\Program Files\\
                                   Tesseract-OCR\\tesseract.exe",
                                   poppler_path = "C:\\Users\\shakhansho\\
                                   Downloads\\Release-22.01.0-0\\
                                   poppler-22.01.0\\Library\\bin")
        >>> images = anonym.pdf2images()
        >>> text = anonym.images2text(images)

        Passing a list of strings to the function

        >>> orgs = anonym.find_ORG(text)
        """
        self._find_EOI(text, EOI='ORG')

    def find_PER(self, text: List[str]) -> Dict[str, List[Tuple[int]]]:
        """
        Find people's names within the string and return the coordinates.

        Wrapper for `_find_EOI` function with the argument `EOI = "PER"`.

        Parameters:
        ----------
        text: List[str]
            A list of strings to search for EOI.

        Returns:
        ----------
        Dict[str, List[Tuple[int, int, int, int]]]
            A dictionary is returned with page numbers as a key and a tuple
            storing 4 integer coordinates as a value.

        Examples
        ----------
        >>> from anonympy.pdf import pdfAnonymizer
        >>> anonym = pdfAnonymizer(path_to_pdf = "Downloads\\test.pdf",
                                   pytesseract_path = "C:\\Program Files\\
                                   Tesseract-OCR\\tesseract.exe",
                                   poppler_path = "C:\\Users\\shakhansho\\
                                   Downloads\\Release-22.01.0-0\\
                                   poppler-22.01.0\\Library\\bin")
        >>> images = anonym.pdf2images()
        >>> text = anonym.images2text(images)

        Passing a list of strings to the function

        >>> names = anonym.find_PER(text)
        """
        return self._find_EOI(text, EOI='PER')

    def find_LOC(self, text: Union[str, list]) -> Dict[str, List[Tuple[int]]]:
        """
        Find organization's names within the string and return the coordinates.

        Wrapper for `_find_EOI` function with the argument `EOI = "LOC"`.

        Parameters:
        ----------
        text: List[str]
            A list of strings to search for EOI.

        Returns:
        ----------
        Dict[str, List[Tuple[int, int, int, int]]]
            A dictionary is returned with page numbers as a key and a tuple
            storing 4 integer coordinates as a value.

        Examples
        ----------
        >>> from anonympy.pdf import pdfAnonymizer
        >>> anonym = pdfAnonymizer(path_to_pdf = "Downloads\\test.pdf",
                                   pytesseract_path = "C:\\Program Files\\
                                   Tesseract-OCR\\tesseract.exe",
                                   poppler_path = "C:\\Users\\shakhansho\\
                                   Downloads\\Release-22.01.0-0\\
                                   poppler-22.01.0\\Library\\bin")
        >>> images = anonym.pdf2images()
        >>> text = anonym.images2text(images)

        Passing a list of strings to the function

        >>> locs = anonym.find_LOC(text)
        """
        return self._find_EOI(text, EOI='LOC')

    def cover_box(self,
                  image: Image.Image,
                  bbox: List[Tuple[int]],
                  fill="black",
                  outline="black") -> None:
        """
        Draw boxes on ROI to hide sensetive information.

        Wrapper `draw_black_box_pytesseract` function from `anonympy.pdf.utils`
        module.

        Parameters:
        ----------
        image: Image.Image
            PIL Image to draw the boxes on

        bbox: List[Tuple[int]]
            Coordinates where to draw

        fill: str, default 'black'
            The color to fill the boxes when covering sensetive information.

        outline: str, default 'black'
            Border color of the boxes.

        Returns:
        ----------
        None
            Changes will be applied directly to the PIL.Image that was passed

        Examples
        ----------
        >>> from anonympy.pdf import pdfAnonymizer
        >>> anonym = pdfAnonymizer(path_to_pdf = "Downloads\\test.pdf",
                                   pytesseract_path = "C:\\Program Files\\
                                   Tesseract-OCR\\tesseract.exe",
                                   poppler_path = "C:\\Users\\shakhansho\\
                                   Downloads\\Release-22.01.0-0\\
                                   poppler-22.01.0\\Library\\bin")
        >>> images = anonym.pdf2images()
        >>> text = anonym.images2text(images)
        >>> numbers = anonym.find_LOC(text)

        Extract the coordinates

        >>> coords = []
        >>> for page, data in numbers.items():
                coords += data

        Passing the coordinates (`List[Tuple[int,int,int,int]]``)
        to the function

        >>> for idx in range(anonym.page_number):
                anonym.cover_box(images[idx], coords[idx])
        """
        draw_black_box_pytesseract(bbox=bbox,
                                   image=image,
                                   fill=fill,
                                   outline=outline)
