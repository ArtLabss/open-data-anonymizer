import pytest
from anonympy.pdf import pdfAnonymizer
from anonympy.pdf.utils_pdf import find_EOI

@pytest.fixture(scope="module")
def anonym_pdf():
    '''
    Initialize `pdfAnonymizer` object
    '''
    url = 'https://raw.githubusercontent.com/ArtLabss/open-data-anonymizer'\
          '/pdfAnonymizer/anonympy/tests/pdf/expected/test.pdf'
    try:
        anonym = pdfAnonymizer(url=url)
    except:  # noqa: E722
        anonym = None

    return anonym


def test_anonym_pdf(anonym_pdf):
    if anonym_pdf is None:
        assert False, "Failed to initialize `pdfAnonymizer` object with "\
                      "`anonympy/tests/pdf/expected/test.pdf` file"
    assert isinstance(anonym_pdf, pdfAnonymizer), "Didn't return "\
                                                  "`pdfAnonymizer` object`"


def test_find_EOI()
