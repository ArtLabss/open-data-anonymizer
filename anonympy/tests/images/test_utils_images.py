import cv2
import urllib
import pytest
import numpy as np
from anonympy import __version__
from anonympy.images.utils_images import find_middle, find_radius, sap_noise
from anonympy.images.utils_images import pixelated


def fetch_image(url):
    req = urllib.request.urlopen(url)
    arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img


@pytest.fixture(scope='module')
def rectangle():
    x, y, w, h = 2, 7, 8, 1
    return x, y, w, h


@pytest.fixture(scope='module')
def load_image():
    # img = fetch_image(
    #     'https://raw.githubusercontent.com/ArtLabss/'
    #     'open-data-anonymizer/main/anonympy/tests/images/expected/sad_boy.jpg')
    img2 = cv2.imread('anonympy/tests/images/expected/sad_boy.jpg')
    return img2


def test_find_middle(rectangle):
    x, y, w, h = rectangle

    output = find_middle(x, y, w, h)
    expected = (6, 7)

    assert output == expected, "`find_middle` returned unexpected value for \
    the following arguments: `x=2, y=7, w=8, h=1`"


def test_find_radius(rectangle):
    x, y, w, h = rectangle

    output = find_radius(x, y, w, h)
    expected = 4

    assert output == expected, "`find_radius` returned unexpected value for \
    the following arguments: `x=2, y=7, w=8, h=1`"


@pytest.mark.skipif(__version__ == '0.2.4',
                    reason="Requires anonympy >= 0.2.5")
def test_sap_noise(load_image):
    output = sap_noise(load_image, seed=42)
    expected = fetch_image(
        'https://raw.githubusercontent.com/ArtLabss/'
        'open-data-anonymizer/main/anonympy/tests/images/expected/'
        'sap_noise.png')
    assert np.array_equal(output, expected), "Image returned by `sap_noise` is\
    different from expected/sap_noise.png"


def test_pixelated(load_image):
    output = pixelated(load_image)
    expected = fetch_image(
        'https://raw.githubusercontent.com/ArtLabss/open-'
        'data-anonymizer/main/anonympy/tests/images/expected/pixelated.png')
    assert np.array_equal(output, expected), "Image returned by `pixelated` is\
    different from expected/pixelated.png"
