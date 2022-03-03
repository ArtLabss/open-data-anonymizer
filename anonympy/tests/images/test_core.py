import os
import cv2
import urllib
import pytest
import numpy as np

from anonympy import __version__
from anonympy.images import imAnonymizer


##def fetch_image(url):
##    req = urllib.request.urlopen(url)
##    arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
##    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
##    return img
##
##
##@pytest.fixture(scope = 'module')
##def load_image():
##    try: 
##        image = fetch_image('https://raw.githubusercontent.com/ArtLabss/open-data-anonymizer/main/anonympy/tests/images/expected/sad_boy.jpg')
##    except urllib.error.HTTPError as eror:
##        image = None
##    return image

##def test_load_image(load_image):
##    if load_image is None:
##        assert False, "Failed to fetch the sample image from `anonympy/tests/images/expected/sad_boy.jpg`"
##    assert True 


def is_similar(image1, image2):
    return image1.shape == image2.shape and not(np.bitwise_xor(image1,image2).any())


##@pytest.fixture(scope = 'module')
def load_image(fname):
    '''
    Load a sample image
    '''
    path = os.path.join(os.path.split(__file__)[0], 'expected', fname)
    img = cv2.imread(path)
    return img


@pytest.fixture(scope="module")
def anonym_img():
    '''
    Initialize `imAnonymizer` object
    '''
    path = os.path.join(os.path.split(__file__)[0], 'expected\\sad_boy.jpg')
    img  = cv2.imread(path)
    if img is None:
        anonym = None
    else:
        anonym = imAnonymizer(img)
    return anonym


def test_anonym_img(anonym_img):
    if anonym_img is None:
        assert False, "Failed to fetch the sample image from `anonympy/tests/images/expected/sad_boy.jpg`"
    assert isinstance(anonym_img, imAnonymizer), "should have returned `imAnonymizer` object"

    
def test_face_blur(anonym_img):
    output = anonym_img.face_blur(shape = 'c', box = None)
    expected = load_image('c_none.png')
    assert is_similar(output, expected), "Image returned by `face_blur(shape = 'c', box = None)` is different from `expected/c_none.png` image"

    output = anonym_img.face_blur(shape = 'r', box = None)
    expected = load_image('r_none.png')
    assert is_similar(output, expected), "Image returned by `face_blur(shape = 'r', box = None)` is different from `expected/r_none.png` image"

    output = anonym_img.face_blur(shape = 'c', box = 'c')
    expected = load_image('cc.png')
    assert is_similar(output, expected), "Image returned by `face_blur(shape = 'c', box = 'c')` is different from `expected/cc.png` image"

    output = anonym_img.face_blur(shape = 'c', box = 'r')
    expected = load_image('cr.png')
    assert is_similar(output, expected), "Image returned by `face_blur(shape = 'c', box = 'r')` is different from `expected/cr.png` image"

    output = anonym_img.face_blur(shape = 'r', box = 'c')
    expected = load_image('rc.png')
    assert is_similar(output, expected), "Image returned by `face_blur(shape = 'r', box = 'c')` is different from `expected/rc.png` image"

    output = anonym_img.face_blur(shape = 'r', box = 'r')
    expected = load_image('rr.png')
    assert is_similar(output, expected), "Image returned by `face_blur(shape = 'r', box = 'r')` is different from `expected/rr.png` image"


@pytest.mark.skipif(__version__ == '0.2.4', reason="Requires anonympy >= 0.2.5 ")
def test_face_SaP(anonym_img):
    output = anonym_img.face_SaP(shape = 'c', box = None, seed = 2)
    expected = load_image('sap_c_none.png')
    assert is_similar(output, expected), "Image returned by `face_SaP(shape = 'c', box = None)` is different from `expected/sap_c_none.png` image"

    output = anonym_img.face_SaP(shape = 'r', box = None, seed = 2)
    expected = load_image('sap_r_none.png')
    assert is_similar(output, expected), "Image returned by `face_SaP(shape = 'r', box = None)` is different from `expected/sap_r_none.png` image"

    output = anonym_img.face_SaP(shape = 'c', box = 'c', seed = 2)
    expected = load_image('sap_c_c.png')
    assert is_similar(output, expected), "Image returned by `face_SaP(shape = 'c', box = 'c')` is different from `expected/sap_c_c.png` image"

    output = anonym_img.face_SaP(shape = 'c', box = 'r', seed = 2)
    expected = load_image('sap_c_r.png')
    assert is_similar(output, expected), "Image returned by `face_SaP(shape = 'c', box = 'r')` is different from `expected/sap_c_r.png` image"

    output = anonym_img.face_SaP(shape = 'r', box = 'c', seed = 2)
    expected = load_image('sap_r_c.png')
    assert is_similar(output, expected), "Image returned by `face_SaP(shape = 'r', box = 'c')` is different from `expected/sap_r_c.png` image"    

    output = anonym_img.face_SaP(shape = 'r', box = 'r', seed = 2)
    expected = load_image('sap_r_r.png')
    assert is_similar(output, expected), "Image returned by `face_SaP(shape = 'r', box = 'r')` is different from `expected/sap_r_r.png` image"
    

def test_face_pixel(anonym_img):
    
    
# from utils import ...
# face_pixel supports only 'r' 


# 'c' None
# 'r' None
# 'c' 'c'
# 'c' 'r'
# 'r' 'c'
# 'r' 'r' 


##assert np.array_equal(output, expected), "Image returned by `sap_noise` is different from expected/sap_noise.png"
