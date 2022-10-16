import cv2
import urllib
import pytest
import numpy as np
from anonympy import __version__
from anonympy.images import imAnonymizer


def is_similar(image1, image2):
    return (image1.shape == image2.shape) and \
            not (np.bitwise_xor(image1, image2).any())


def fetch_image(url):
    req = urllib.request.urlopen(url)
    arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img


def load_image(fname):
    img = fetch_image(
        'https://raw.githubusercontent.com/ArtLabss/'
        f'open-data-anonymizer/main/anonympy/tests/images/expected/{fname}')
    return img


@pytest.fixture(scope="module")
def anonym_img():
    '''
    Initialize `imAnonymizer` object
    '''
    img = fetch_image(
        'https://raw.githubusercontent.com/ArtLabss/'
        'open-data-anonymizer/main/anonympy/tests/images/expected/sad_boy.jpg')
    if img is None:
        anonym = None
    else:
        anonym = imAnonymizer(img)
    return anonym


def test_anonym_img(anonym_img):
    if anonym_img is None:
        assert False, "Failed to fetch the sample image from"\
                      "`anonympy/tests/images/expected/sad_boy.jpg`"
    assert isinstance(anonym_img, imAnonymizer), "Didn't return " \
                                                 "`imAnonymizer` object"


def test_face_blur(anonym_img):
    output = anonym_img.face_blur(shape='c', box=None)
    expected = load_image('c_none.png')
    assert is_similar(output, expected), "Image returned by `face_blur(shape =\
     'c', box = None)` is different from `expected/c_none.png` image"

    output = anonym_img.face_blur(shape='r', box=None)
    expected = load_image('r_none.png')
    assert is_similar(output, expected), "Image returned by `face_blur(shape =\
    'r', box = None)` is different from `expected/r_none.png` image"

    output = anonym_img.face_blur(shape='c', box='c')
    expected = load_image('cc.png')
    assert is_similar(output, expected), "Image returned by `face_blur(shape =\
    'c', box = 'c')` is different from `expected/cc.png` image"

    output = anonym_img.face_blur(shape='c', box='r')
    expected = load_image('cr.png')
    assert is_similar(output, expected), "Image returned by `face_blur(shape =\
    'c', box = 'r')` is different from `expected/cr.png` image"

    output = anonym_img.face_blur(shape='r', box='c')
    expected = load_image('rc.png')
    assert is_similar(output, expected), "Image returned by `face_blur(shape =\
    'r', box = 'c')` is different from `expected/rc.png` image"

    output = anonym_img.face_blur(shape='r', box='r')
    expected = load_image('rr.png')
    assert is_similar(output, expected), "Image returned by `face_blur(shape =\
    'r', box = 'r')` is different from `expected/rr.png` image"


@pytest.mark.skipif(__version__ == '0.2.4',
                    reason="Requires anonympy >= 0.2.5")
def test_face_SaP(anonym_img):
    output = anonym_img.face_SaP(shape='c', box=None, seed=2)
    expected = load_image('sap_c_none.png')
    assert is_similar(output, expected), "Image returned by `face_SaP(shape =\
    'c', box = None)` is different from `expected/sap_c_none.png` image"

    output = anonym_img.face_SaP(shape='r', box=None, seed=2)
    expected = load_image('sap_r_none.png')
    assert is_similar(output, expected), "Image returned by `face_SaP(shape =\
    'r', box = None)` is different from `expected/sap_r_none.png` image"

    output = anonym_img.face_SaP(shape='c', box='c', seed=2)
    expected = load_image('sap_c_c.png')
    assert is_similar(output, expected), "Image returned by `face_SaP(shape =\
    'c', box = 'c')` is different from `expected/sap_c_c.png` image"

    output = anonym_img.face_SaP(shape='c', box='r', seed=2)
    expected = load_image('sap_c_r.png')
    assert is_similar(output, expected), "Image returned by `face_SaP(shape =\
    'c', box = 'r')` is different from `expected/sap_c_r.png` image"

    output = anonym_img.face_SaP(shape='r', box='c', seed=2)
    expected = load_image('sap_r_c.png')
    assert is_similar(output, expected), "Image returned by `face_SaP(shape =\
    'r', box = 'c')` is different from `expected/sap_r_c.png` image"

    output = anonym_img.face_SaP(shape='r', box='r', seed=2)
    expected = load_image('sap_r_r.png')
    assert is_similar(output, expected), "Image returned by `face_SaP(shape =\
    'r', box = 'r')` is different from `expected/sap_r_r.png` image"


@pytest.mark.skipif(__version__ == '0.2.4',
                    reason="Requires anonympy >= 0.2.5")
def test_face_pixel(anonym_img):
    output = anonym_img.face_pixel(blocks=20, box=None)
    expected = load_image('pixel_none.png')
    assert is_similar(output, expected), "Image returned by `face_pixel(blocks\
     = 20, box = None)` is different from `expected/pixel_none.png` image"

    output = anonym_img.face_pixel(blocks=50, box=None)
    expected = load_image('pixel_50_none.png')
    assert is_similar(output, expected), "Image returned by `face_pixel(blocks\
     = 50, box = None)` is different from `expected/pixel_50_none.png` image"

    output = anonym_img.face_pixel(blocks=20, box='r')
    expected = load_image('pixel_r.png')
    assert is_similar(output, expected), "Image returned by `face_pixel(blocks\
     = 20, box = 'r')` is different from `expected/pixel_r.png` image"

    output = anonym_img.face_pixel(blocks=50, box='r')
    expected = load_image('pixel_50_r.png')
    assert is_similar(output, expected), "Image returned by `face_pixel(blocks\
     = 50, box = 'r')` is different from `expected/pixel_50_r.png` image"


def test_blur(anonym_img):
    output = anonym_img.blur()
    expected = load_image('gaussian_blur.png')
    assert is_similar(output, expected), "Image returned by \
    `blur(method='gaussian', kernel=(15,15))` is different from \
    `expected/gaussian_blur.png` image"

    output = anonym_img.blur('averaging', kernel=(30, 30))
    expected = load_image('averaging_blur.png')
    assert is_similar(output, expected), "Image returned by `blur('averaging',\
     kernel=(30,30))` is different from `expected/averaging_blur.png` image"

    output = anonym_img.blur('bilateral', kernel=(30, 150, 150))
    expected = load_image('bilateral_blur.png')
    assert is_similar(output, expected), "Image returned by `blur('bilateral',\
     kernel=(30, 150, 150))` is different \
     from `expected/bilateral_blur.png` image"

    output = anonym_img.blur('median', kernel=11)
    expected = load_image('median_blur.png')
    assert is_similar(output, expected), "Image returned by `blur('median',\
    kernel=11)` is different from `expected/median_blur.png` image"
