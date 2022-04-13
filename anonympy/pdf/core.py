import re
import os 
import cv2
import numpy as np
import pytesseract
from typing import List, Union, Tuple, Dict
from pdf2image import convert_from_path

from utils import * # from anonympy.pdf.utils import *

import PIL
from PIL import Image

from transformers import pipeline
from transformers import AutoTokenizer
from transformers import AutoModelForTokenClassification


class pdfAnonymizer(object):
    """
    Initializes pdfAnonymizer object

    Parameters:
    ----------
         
    Returns:
    ----------

    Raises
    ----------

    See also
    ----------

    Examples
    ----------
    """
    def __init__(self, 
        path_to_pdf: str,
        # path_to_folder: str = None,
        pytesseract_path: str = None,
        poppler_path: str = None,
        model: str = "dbmdz/bert-large-cased-finetuned-conll03-english",
        tokenizer: str = "dbmdz/bert-large-cased-finetuned-conll03-english"):

        if path_to_pdf is not None:
            if not os.path.exists(path_to_pdf):
                raise Exception(f"Can't find PDF file at `{path_to_pdf}`")
            elif not path_to_pdf.endswith('.pdf'):
                raise Exception(f"`String path should end with `.pdf`. But {path_to_pdf[-4:]} was found.")
        
        # elif path_to_folder is not None:
        #     if not os.path.isdir(path_to_folder):
        #         raise Exception(f"Can't find folder at `{path_to_folder}`")
        #     else:
        #         for file in os.listdir(path_to_folder):
        #             if not file.endswith('.pdf'):
        #                 raise Exception(f"`Folder should contain only `.pdf` files. But {file[-4:]} was found.")

        if pytesseract_path is not None:
            pytesseract.pytesseract.tesseract_cmd = pytesseract_path

        if poppler_path is not None:
            self.poppler_path = poppler_path

        self.nlp = pipeline("ner", aggregation_strategy="simple", model=model, tokenizer=tokenizer)
        self.path_to_pdf = path_to_pdf
#         self.path_to_folder = path_to_folder 
        self.number_of_pages  = None # changes in `pdfAnonymizer.pdf2images`
        self.images = [] # anonymized images that will be converted to PDF
        self.bbox = [] # bounding boxes of PII_objects 
        self.pages_data = [] # data that will be returned by pytesseract (pdfAnonymizer.images2text)
        self.PII_objects = [] # Personal Identifiable Information 


    def anonymize(self, output_path: str, to_pdf: bool=True, remove_metadata: bool=True, fill: str='black', outline: str='black') -> None:
        """
        Master function for PDF anonymization. 
        PDF to Image -> Image to Text -> NER on Text -> Black Box str if classified as PII

        Parameters:
        ----------
             
        Returns:
        ----------

        Raises
        ----------

        Examples
        ----------
        """
        if not output_path.endswith('.pdf'):
                raise Exception(f"`String path should end with `.pdf`. But {output_path[-4:]} was found.")

        self.images += self.pdf2images()
        page_number = 1
        
        if self.number_of_pages == 1: # 1 Page PDF 
            text = self.images2text(self.images)[0] # return str if 1 page PDF, else a list of str
            ner = self.nlp(text)

            find_emails(text = text, matches = self.PII_objects)
            find_numbers(text = text, matches = self.PII_objects)
            find_months(text = text, matches = self.PII_objects)

            find_EOI(pipeline=ner, matches=self.PII_objects, EOI="PER")
            find_EOI(pipeline=ner, matches=self.PII_objects, EOI="ORG")
            find_EOI(pipeline=ner, matches=self.PII_objects, EOI="LOC")

            find_coordinates_pytesseract(matches=self.PII_objects, data=self.pages_data[page_number - 1], bbox=self.bbox)

            self.cover_box(image=self.images[0], bbox=self.bbox, fill=fill, outline=outline)        
        else: # more than 1 page 
            text = self.images2text(self.images)

            for excerpt in text:
                temp_pii = []
                temp_bbox = []
                ner = self.nlp(excerpt)

                find_emails(text = excerpt, matches = temp_pii)
                find_numbers(text = excerpt, matches = temp_pii)
                find_months(text = excerpt, matches = temp_pii)

                find_EOI(pipeline=ner, matches=temp_pii, EOI="PER")
                find_EOI(pipeline=ner, matches=temp_pii, EOI="ORG")
                find_EOI(pipeline=ner, matches=temp_pii, EOI="LOC")

                find_coordinates_pytesseract(matches=temp_pii, data=self.pages_data[page_number - 1], bbox=temp_bbox)
                self.cover_box(self.images[page_number - 1], temp_bbox, fill=fill, outline=outline)

                self.PII_objects.append({f'page_{page_number}': temp_pii})
                self.bbox.append({f'page_{page_number}': temp_bbox})

                page_number += 1

        if to_pdf: 
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
                    img1.save(output_path, save_all=True, append_images=self.images)
    

    def pdf2images(self) -> List[Image.Image]:
        """
        Convert PDF file to a list of images.
        Based on `pdf2image` library

        Parameters:
        ----------

        Returns:
        ----------

        Raises
        ----------

        Examples
        ----------
        """
        if self.poppler_path is None:
            images = convert_from_path(self.path_to_pdf)
        else:
            images = convert_from_path(self.path_to_pdf, poppler_path = self.poppler_path)
        
        self.number_of_pages = len(images)
        return images


    def images2text(self, images: List[Image.Image]) -> List[str]:
        """
        Extract text from an image.
        Based on `pytesseract.image_to_data`

        Parameters:
        ----------
             
        Returns:
        ----------

        Raises
        ----------

        Examples
        ----------

        """

        if len(images) == 1:

            page = pytesseract.image_to_data(images[0], output_type = "dict")
            self.pages_data.append(page)
            excerpt = ""

            for line in page["text"]:
                if line == "":
                    pass
                else:
                    excerpt += line.strip() + " "

            return [excerpt]
        
        else:
            excerpts = []

            for image in images:
                page = pytesseract.image_to_data(image, output_type = "dict")
                self.pages_data.append(page)
                excerpt = ""

                for line in page["text"]:
                    if line == "":
                        pass
                    else:
                        excerpt += line.strip() + " "

                excerpts.append(excerpt)

            return excerpts


    def find_emails(self, text: Union[str, List[str]]) -> dict[str, List[Tuple[int, int, int, int]]]:
        """
        Find emails within the string and return the coordinates.
        Based on `find_emails` function `from anonympy.pdf.utils`

        Parameters:
        ----------

        Returns:
        ----------

        Raises
        ----------

        Examples
        ----------

        """
        coords = {}
        page_number = 1

        if type(text) is str:
            emails = []
            bbox = []

            find_emails(text, emails)
            find_coordinates_pytesseract(emails, self.pages_data[page_number-1], bbox)
            coords[f'page_{page_number}'] = bbox
            return coords 

        else:
            for excerpt in text:
                emails = []
                bbox = []

                find_emails(excerpt, emails)
                find_coordinates_pytesseract(emails, self.pages_data[page_number-1], bbox)
                coords[f'page_{page_number}'] = bbox
                page_number += 1
            return coords


    def find_numbers(self, text: Union[str, list]) -> dict[str, List[Tuple[int, int, int, int]]]:        
        """
        Find numbers within the string and return the coordinates.
        Based on `find_numbers` function `from anonympy.pdf.utils`

        Parameters:
        ----------

        Returns:
        ----------

        Raises
        ----------

        Examples
        ----------
        """
        coords = {}
        page_number = 1

        if type(text) is str:
            bbox = []
            numbers = []

            find_numbers(text, numbers)
            find_coordinates_pytesseract(numbers, self.pages_data[page_number-1], bbox)
            coords[f'page_{page_number}'] = bbox
            return coords 

        else:
            for excerpt in text:
                bbox = []
                numbers = []

                find_numbers(excerpt, numbers)
                find_coordinates_pytesseract(numbers, self.pages_data[page_number-1], bbox)
                coords[f'page_{page_number}'] = bbox
                page_number += 1
            return coords


    def find_months(self, text: str) -> dict[str, List[Tuple[int, int, int, int]]]:
        """
        Find months within the string and return the coordinates.
        Based on `find_months` function `from anonympy.pdf.utils`

        Parameters:
        ----------

        Returns:
        ----------

        Raises
        ----------

        Examples
        ----------

        """
        months = []
        bbox = []
        coords = {}
        page_number = 1

        if type(text) is str:
            find_months(text, months)
            find_coordinates_pytesseract(months, self.pages_data[page_number-1], bbox)
            coords[f'page_{page_number}'] = bbox
            return coords 

        else:
            for excerpt in text:
                months = []
                bbox = []

                find_months(excerpt, months)
                find_coordinates_pytesseract(months, self.pages_data[page_number-1], bbox)
                coords[f'page_{page_number}'] = bbox
                page_number += 1
            return coords


    def _find_EOI(self, text: Union[str, list], EOI: str) -> dict[str, List[Tuple[int, int, int, int]]]:
        """
        Find EOI (Entity of Interest) within the string and return the coordinates.
        Based on `find_EOI` function `from anonympy.pdf.utils`

        Parameters:
        ----------

        Returns:
        ----------

        Raises
        ----------

        Examples
        ----------

        """
        coords = {}
        page_number = 1 

        if type(text) is str:
            ner = self.nlp(text)
            bbox = []
            names = []

            find_EOI(ner, names, EOI) 
            find_coordinates_pytesseract(names, self.pages_data[page_number - 1], bbox)
            coords[f'page_{page_number}'] = bbox
            return coords

        else:
            for excerpt in text:
                ner = self.nlp(excerpt)
                names = []
                bbox = []

                find_EOI(ner, names, EOI)
                find_coordinates_pytesseract(names, self.pages_data[page_number-1], bbox)
                coords[f'page_{page_number}'] = bbox
                page_number += 1
            return coords


    def find_ORG(self, text: Union[str, list]) -> dict[str, List[Tuple[int, int, int, int]]]:
        """
        Find organization's names within the string and return the coordinates.
        Based on `find_EOI` function `from anonympy.pdf.utils`

        Parameters:
        ----------

        Returns:
        ----------

        Raises
        ----------

        Examples
        ----------

        """
        self._find_EOI(text, EOI='ORG')


    def find_PER(self, text: Union[str, list]) -> dict[str, List[Tuple[int, int, int, int]]]:
        """
        Find organization's names within the string and return the coordinates.
        Based on `find_EOI` function `from anonympy.pdf.utils`

        Parameters:
        ----------

        Returns:
        ----------

        Raises
        ----------

        Examples
        ----------

        """
        return self._find_EOI(text, EOI='PER')


    def find_LOC(self, text: Union[str, list]) -> dict[str, List[Tuple[int, int, int, int]]]:
        """
        Find organization's names within the string and return the coordinates.
        Based on `find_EOI` function `from anonympy.pdf.utils`

        Parameters:
        ----------

        Returns:
        ----------

        Raises
        ----------

        Examples
        ----------

        """
        return self._find_EOI(text, EOI='LOC')


    def cover_box(self, image: Image.Image, bbox: List[Tuple[int]],  fill="black", outline="black") -> None:
        """
        Draw boxes on ROI to hide sensetive information (default, black box).  
        Based on `draw_black_box_pytesseract` function `from anonympy.pdf.utils`

        Parameters:
        ----------

        Returns:
        ----------

        Raises
        ----------

        Examples
        ----------

        """
        draw_black_box_pytesseract(bbox=bbox,image=image,fill=fill, outline=outline)