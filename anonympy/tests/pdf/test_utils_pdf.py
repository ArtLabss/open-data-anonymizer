#  These tests can be run locally, however change
#  `pytesseract_path` and `poppler_path`
#  arguments within `anonym_obj` function


import cv2
import pytest
import urllib
import numpy as np
from PIL import ImageDraw
from anonympy.pdf import pdfAnonymizer
from anonympy.pdf.utils_pdf import draw_black_box_pytesseract, find_EOI
from anonympy.pdf.utils_pdf import find_coordinates_pytesseract
from anonympy.pdf.utils_pdf import find_months, find_emails, find_numbers


def fetch_image(url):
    req = urllib.request.urlopen(url)
    arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img


def is_similar(image1, image2):
    return (image1.size == image2.size) and \
            not (np.bitwise_xor(image1, image2).any())


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


def test_draw_black_box_pytesseract(anonym_obj):
    anonym_obj.pdf2images()
    bbox = [(570, 280, 561, 28)]

    expected = anonym_obj.images[0].copy()
    draw = ImageDraw.Draw(expected)

    for box in bbox:
        x, y, w, h = box
        draw.rectangle([x, y, x + w, y + h], fill='black', outline='black')

    output = anonym_obj.images[0].copy()
    draw_black_box_pytesseract(bbox, output)

    assert is_similar(expected, output), ("Method `cover_box`"
                                          "didn't return expected values")


def test_find_coordinates_pytesseract(anonym_obj):
    anonym_obj.images2text(anonym_obj.images)

    expected = [(570, 280, 561, 28)]
    PII_object = ['shakhansho.sabzaliev_2023@ucentralasia.org']

    output = []
    find_coordinates_pytesseract(PII_object,
                                 anonym_obj.pages_data[0],
                                 output)

    assert output == expected, ('Expected email coordinates (570, 280, 561'
                                f', 28), but function returned {output[0]}')


def test_find_EOI(anonym_obj):
    pipeline = anonym_obj._nlp(anonym_obj.texts[0])

    expected_PER = ['hansho', 'Sabzal', 'Elvira', 'Sagyntay', 'hansh']
    output_PER = []

    find_EOI(pipeline=pipeline, matches=output_PER, EOI='PER')
    assert expected_PER == output_PER, ("`find_EOI` returned unexpected "
                                        "values for `EOI='PER'`")

    expected_LOC = ['Parkovaya', 'Moscow', 'Russia', 'Bishkek', 'Kyrgystan']
    output_LOC = []

    find_EOI(pipeline=pipeline, matches=output_LOC, EOI='LOC')
    assert expected_LOC == output_LOC, ("`find_EOI` returned unexpected "
                                        "values for `EOI='LOC'`")

    expected_ORG = ['Shak', 'ucent', 'CRM', 'Technologies', 'Panphilova',
                    'Inter', 'CRM', 'Technologies']
    output_ORG = []

    find_EOI(pipeline=pipeline, matches=output_ORG, EOI='ORG')
    assert expected_ORG == output_ORG, ("`find_EOI` returned unexpected "
                                        "values for `EOI='ORG'`")


def test_find_emails(anonym_obj):
    expected = ['shakhansho.sabzaliev_2023@ucentralasia.org']
    output = []

    find_emails(anonym_obj.texts[0], output)
    assert expected == output, ("Method `find_emails`"
                                "didn't return expected values")


def test_find_months(anonym_obj):
    expected = ['November', 'November']
    output = []

    find_months(anonym_obj.texts[0], output)
    assert expected == output, ("Method `find_months`"
                                "didn't return expected values")


def test_find_numbers(anonym_obj):
    expected = ['11', '39', '1', '105264', '2023',
                '19', '2020', '1', '18', '2020']
    output = []

    find_numbers(anonym_obj.texts[0], output)
    assert expected == output, ("Method `find_numbers`"
                                "didn't return expected values")
