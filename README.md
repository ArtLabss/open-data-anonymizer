<p align='center'>
  <a href="https://artlabs.tech/">
    <img src='https://raw.githubusercontent.com/ArtLabss/tennis-tracking/main/VideoOutput/artlabs%20logo.jpg' width="150" height="170">
  </a>
</p>
<h1 align='center'>anonympy 🕶️</h1>

<p align='center'>
<img src="https://img.shields.io/github/forks/ArtLabss/open-data-anonimizer.svg">
  <img src="https://img.shields.io/github/stars/ArtLabss/open-data-anonimizer.svg">
  <img src="https://img.shields.io/github/watchers/ArtLabss/open-data-anonimizer.svg">
  <img src="https://img.shields.io/github/last-commit/ArtLabss/open-data-anonimizer.svg">
  <br>
  <img src="https://img.shields.io/pypi/v/anonympy.svg">
  <img src="https://img.shields.io/pypi/l/anonympy.svg">
  <img src="https://hits.sh/github.com/ArtLabss/open-data-anonimizer.svg">
  <a href="https://pepy.tech/project/anonympy"><img src="https://pepy.tech/badge/anonympy"></a>
  <br>
  <a href="https://github.com/ArtLabss/open-data-anonymizer/actions/workflows/pylinter.yml"><img src="https://github.com/ArtLabss/open-data-anonymizer/actions/workflows/pylinter.yml/badge.svg"></a>
  <a href="https://github.com/ArtLabss/open-data-anonymizer/actions/workflows/python-app.yml"><img src="https://github.com/ArtLabss/open-data-anonymizer/actions/workflows/python-app.yml/badge.svg"></a>
  <a href="https://github.com/ArtLabss/open-data-anonymizer/actions/workflows/codeql-analysis.yml"><img src="https://github.com/ArtLabss/open-data-anonymizer/actions/workflows/codeql-analysis.yml/badge.svg"></a>
  <br>
  <code>With ❤️ by ArtLabs</code>
  
<h2>Overview</h2>
<p>General Data Anonymization library for images, PDFs and tabular data. See <a href="https://artlabs.tech/projects/">ArtLabs/projects</a> for more or similar projects.</p>
<br>
<h2>Main Features</h2>

<p>Ease of use - this package was written to be as intuitive as possible.</p>

<p><strong>Tabular</strong></p>
<ul>
  <li>Efficient - based on pd.DataFrame</li>
  <li>Numerous anonymization methods</li>
    <ul>
      <li>Numeric data</li>
        <ul>
          <li>Generalization - Binning</li>
          <li>Perturbation</li>
          <li>PCA Masking</li>
          <li>Generalization - Rounding</li>
        </ul>
      <li>Categorical data</li>
        <ul>
          <li>Synthetic Data</li>
          <li>Resampling</li>
          <li>Tokenization</li>
          <li>Partial Email Masking</li>
        </ul>
      <li>Datetime data</li>
        <ul>
          <li>Synthetic Date</li>
          <li>Perturbation</li>
        </ul>
      </ul>
</ul>

<p><strong>Images</strong></p>
<ul>
  <li>Anonymization techniques</li>
  <ul>
    <li>Personal Images (faces)</li>
    <ul>
      <li>Blurring</li>
      <li>Pixaled Face Blurring</li>
      <li>Salt and Pepper Noise</li>
    </ul>
    <li>General Images</li>
    <ul>
      <li>Blurring</li>
    </ul>
  </ul>
</ul>

<p><strong>PDF</strong></p>
<ul>
  <li>Find sensitive information and cover it with black boxes</li>
</ul>

<p><strong>Text, Sound</strong></p>
<ul>
  <li>In Development</li>
</ul>

<br>

<h2>Installation</h2>

<h3>Dependencies</h3>
<ol>
  <li> Python (>= 3.7)</li>
  <li>cape-privacy</li>
  <li>faker</li>
  <li>pandas</li>
  <li>OpenCV</li>
  <li>pytesseract</li>
  <li>transformers</li>
  <li><a href="https://github.com/ArtLabss/open-data-anonimizer/blob/main/requirements.txt">.         .  .  .  .  </a></li>
</ol>

<h3>Install with pip</h3>

<p>Easiest way to install anonympy is using <code>pip</code></p>

```
pip install anonympy
```
<p>Due to conflicting pandas/numpy versions with <a href="https://github.com/capeprivacy/cape-python/issues/112">cape-privacy</a>, it's recommend to install them seperately</p>

```
pip install cape-privacy==0.3.0 --no-deps 
```

<h3>Install from source</h3>

<p>Installing the library from source code is also possible</p>

```
git clone https://github.com/ArtLabss/open-data-anonimizer.git
cd open-data-anonimizer
pip install -r requirements.txt
make bootstrap
pip install cape-privacy==0.3.0 --no-deps 
```

<h3>Downloading Repository</h3>

<p>Or you could download this repository from <a href="https://pypi.org/project/anonympy/">pypi</a> and run the following:

```
cd open-data-anonimizer
python setup.py install
```


<br>

<h2>Usage Example </h2>

[![Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1wg4g4xWTSLvThYHYLKDIKSJEC4ChQHaM?usp=sharing)

<p>More examples <a href="https://github.com/ArtLabss/open-data-anonimizer/blob/b5d5f2df94b80011a8a93fa08f0046d1390cec49/examples/examples.ipynb">here</a>
  
<p><strong>Tabular</strong></p>

```python
>>> from anonympy.pandas import dfAnonymizer
>>> from anonympy.pandas.utils_pandas import load_dataset

>>> df = load_dataset() 
>>> print(df)
```

|   |  name | age |  birthdate |   salary |                                  web |                email |       ssn |
|--:|------:|----:|-----------:|---------:|-------------------------------------:|---------------------:|----------:|
| 0 | Bruce | 33  | 1915-04-17 | 59234.32 | http://www.alandrosenburgcpapc.co.uk | josefrazier@owen.com | 343554334 |
| 1 | Tony  | 48  | 1970-05-29 | 49324.53 | http://www.capgeminiamerica.co.uk    | eryan@lewis.com      | 656564664 |
  
```python
# Calling the generic function
>>> anonym = dfAnonymizer(df)
>>> anonym.anonymize(inplace = False) # changes will be returned, not applied
```

|      | name            | age    | birthdate  | age     | web        |         email       |     ssn     |
|------|-----------------|--------|------------|---------|------------|---------------------|-------------|
| 0    | Stephanie Patel | 30     | 1915-05-10 | 60000.0 | 5968b7880f | pjordan@example.com | 391-77-9210 |
| 1    | Daniel Matthews | 50     | 1971-01-21 | 50000.0 | 2ae31d40d4 | tparks@example.org  | 872-80-9114 |
  
```python
# Or applying a specific anonymization technique to a column
>>> from anonympy.pandas.utils_pandas import available_methods

>>> anonym.categorical_columns
... ['name', 'web', 'email', 'ssn']
>>> available_methods('categorical') 
... categorical_fake	categorical_fake_auto	categorical_resampling	categorical_tokenization	categorical_email_masking

>>> anonym.anonymize({'name': 'categorical_fake',  # {'column_name': 'method_name'}
                  'age': 'numeric_noise',
                  'birthdate': 'datetime_noise',
                  'salary': 'numeric_rounding',
                  'web': 'categorical_tokenization', 
                  'email':'categorical_email_masking', 
                  'ssn': 'column_suppression'})
>>> print(anonym.to_df())
```
|   |  name | age |  birthdate |   salary |                                  web |                email |
|--:|------:|----:|-----------:|---------:|-------------------------------------:|---------------------:|
| 0 | Paul Lang | 31  | 1915-04-17 | 60000.0 | 8ee92fb1bd | j*****r@owen.com |
| 1 | Michael Gillespie  | 42  | 1970-05-29 | 50000.0 | 51b615c92e    | e*****n@lewis.com      | 
 
<br >
<p><strong>Images</strong></p>

```python
# Passing an Image
>>> import cv2
>>> from anonympy.images import imAnonymizer

>>> img = cv2.imread('salty.jpg')
>>> anonym = imAnonymizer(img)

>>> blurred = anonym.face_blur((31, 31), shape='r', box = 'r')  # blurring shape and bounding box ('r' / 'c')
>>> pixel = anonym.face_pixel(blocks=20, box=None)
>>> sap = anonym.face_SaP(shape = 'c', box=None)
```
blurred            |  pixel           |    sap
:-------------------------:|:-------------------------:|:-------------------------:
![input_img1](https://raw.githubusercontent.com/ArtLabss/open-data-anonimizer/d61127f7a8fdff603af21dcab8edbf72f2aab292/examples/files/sad_boy_blurred.jpg)  |  ![output_img1](https://raw.githubusercontent.com/ArtLabss/open-data-anonimizer/d61127f7a8fdff603af21dcab8edbf72f2aab292/examples/files/sad_boy_pixel.jpg)    |   ![sap_image](https://raw.githubusercontent.com/ArtLabss/open-data-anonimizer/d61127f7a8fdff603af21dcab8edbf72f2aab292/examples/files/sad_boy_sap.jpg) 

```python
# Passing a Folder 
>>> path = 'C:/Users/shakhansho.sabzaliev/Downloads/Data' # images are inside `Data` folder
>>> dst = 'D:/' # destination folder
>>> anonym = imAnonymizer(path, dst)

>>> anonym.blur(method = 'median', kernel = 11) 
```

<p>This will create a folder <i>Output</i> in <code>dst</code> directory.</p>

```python
# The Data folder had the following structure

|   1.jpg
|   2.jpg
|   3.jpeg
|   
\---test
    |   4.png
    |   5.jpeg
    |   
    \---test2
            6.png

# The Output folder will have the same structure and file names but blurred images
```

<br>

<p><strong>PDF</strong></p>

<p>In order to initialize <code>pdfAnonymizer</code> object we have to install <code>pytesseract</code> and <code>poppler</code>, and provide path to the binaries of both as arguments or add paths to system variables</p>

```python
>>> from anonympy.pdf import pdfAnonymizer

# need to specify paths, since I don't have them in system variables
>>> anonym = pdfAnonymizer(path_to_pdf = "Downloads\\test.pdf",
                       pytesseract_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe",
                       poppler_path = r"C:\Users\shakhansho\Downloads\Release-22.01.0-0\poppler-22.01.0\Library\bin")

# Calling the generic function
>>> anonym.anonymize(output_path = 'output.pdf',
                     remove_metadata = True,
                     fill = 'black',
                     outline = 'black')
```

`test.pdf`            |  `output.pdf`            | 
:-------------------------:|:-------------------------:|
![test_img](https://raw.githubusercontent.com/ArtLabss/open-data-anonymizer/f09e98c05380ffda6cecdd5b332e3dc66a30e17c/examples/files/test-1.jpg)  |  ![output_img](https://raw.githubusercontent.com/ArtLabss/open-data-anonymizer/be3f376e6d93e7a726f083bf28db3bcbd7f592a3/examples/files/test_output.jpg)    |

<p>In case you only want to hide specific information, instead of <code>anonymize</code> use other methods</p>

```python
>>> anonym = pdfAnonymizer(path_to_pdf = r"Downloads\test.pdf")
>>> anonym.pdf2images() #  images are stored in anonym.images variable 
>>> anonym.images2text(anonym.images) # texts are stored in anonym.texts

#  Entities of interest 
>>> locs: dict = anonym.find_LOC(anonym.texts[0])  # index refers to page number
>>> emails: dict = anonym.find_emails(anonym.texts[0])  # {page_number: [coords]}
>>> coords: list = locs['page_1'] + emails['page_1'] 

>>> anonym.cover_box(anonym.images[0], coords)
>>> display(anonym.images[0])
```

<h2>Development</h2>

<h3>Contributions</h3>

<p>The <a href="https://github.com/ArtLabss/open-data-anonimizer/blob/main/CONTRIBUTING.md">Contributing Guide</a> has detailed information about contributing code and documentation.</p>

<h3>Important Links</h3>
<ul>
  <li>Official source code repo: <a href="https://github.com/ArtLabss/open-data-anonimizer">https://github.com/ArtLabss/open-data-anonimizer</a></li>
  <li>Download releases: <a href="https://pypi.org/project/anonympy/">https://pypi.org/project/anonympy/</a></li>
  <li>Issue tracker: <a href="https://github.com/ArtLabss/open-data-anonimizer/issues">https://github.com/ArtLabss/open-data-anonimizer/issues</li></a>
</ul>

<h2>License</h2>

<p><a href="https://github.com/ArtLabss/open-data-anonimizer/blob/main/LICENSE">BSD-3</a></p>


<h2>Code of Conduct</h2>
<p>Please see <a href="https://github.com/ArtLabss/open-data-anonimizer/blob/main/CODE_OF_CONDUCT.md">Code of Conduct</a>. 
All community members are expected to follow it.</p>
