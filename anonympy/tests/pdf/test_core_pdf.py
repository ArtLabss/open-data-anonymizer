# #  These tests can be run locally, however change
# #  `pytesseract_path` and `poppler_path`
# #  arguments within `anonym_obj` function

import cv2
import pytest
import urllib
import numpy as np
from anonympy.pdf import pdfAnonymizer


def fetch_image(url):
    req = urllib.request.urlopen(url)
    arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img


@pytest.fixture(scope="session")
def anonym_obj():
    '''
    Initialize `pdfAnonymizer` object
    '''
    anonym = pdfAnonymizer(
        path_to_pdf='anonympy/tests/pdf/expected/test.pdf',
        model=("dbmdz/bert-large-cased-"
               "finetuned-conll03-english"),
        tokenizer=("dbmdz/bert-large-cased"
                   "-finetuned-conll03-english"))
    return anonym


def test_anonym_obj(anonym_obj):
    if anonym_obj is None:
        assert False, ("Failed to initialize `pdfAnonymizer` object with "
                       "`anonympy/tests/pdf/expected/test.pdf` file")
    assert isinstance(anonym_obj, pdfAnonymizer), ("Expected to return "
                                                   "`pdfAnonymizer` object`")

    anonym_obj.pdf2images()

    assert len(anonym_obj.images) == 1, ("`pdf2images` didn't return"
                                         " expected value")

    anonym_obj.images2text(anonym_obj.images)

    assert len(anonym_obj.texts) == 1, ("`images2text` method didn't"
                                        " return expected value")
    assert type(anonym_obj.texts[0]) is str, ("Expected Type `str`")

    assert anonym_obj.number_of_pages == 1, ('Unexpected value returned')

    assert type(anonym_obj.pages_data[0]) is dict, ("Unexpected value"
                                                    " returned")
