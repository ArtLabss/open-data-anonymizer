<p align='center'>
  <a href="https://artlabs.tech/">
    <img src='https://raw.githubusercontent.com/ArtLabss/open-data-anonimizer/8c15181375a6a6e361aa776ab1ffccd486729d1a/examples/files/artLabs_new.png'>
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
  <code>With ❤️ by ArtLabs</code>
  
<h2>Overview</h2>
<p>A general Python library for data anonymization of tabular, text, image and sound data. See <a href="https://artlabs.tech/projects/">ArtLabs/projects</a> for more or similar projects.</p>
<br>
<h2>Main Features</h2>

<p><strong>Tabular</strong></p>

<ul>
  <li>Ease of use</li>
  <li>Efficient anonymization (based on pandas DataFrame)</li>
  <li>Numerous anonymization techniques</li>
    <ul>
      <li>Numeric</li>
        <ul>
          <li>Generalization - Binning</li>
          <li>Perturbation</li>
          <li>PCA Masking</li>
          <li>Generalization - Rounding</li>
        </ul>
      <li>Categorical</li>
        <ul>
          <li>Synthetic Data</li>
          <li>Resampling</li>
          <li>Tokenization</li>
          <li>Partial Email Masking</li>
        </ul>
      <li>DateTime</li>
        <ul>
          <li>Synthetic Date</li>
          <li>Perturbation</li>
        </ul>
      </ul>
</ul>

<p><strong>Images</strong></p>
<ul>
  <li>Anonymization Techniques</li>
  <ul>
    <li>Personal Images</li>
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
  <li><a href="https://github.com/ArtLabss/open-data-anonimizer/blob/main/requirements.txt">    . . .</a></li>
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

<p>You can find more examples <a href="https://github.com/ArtLabss/open-data-anonimizer/blob/b5d5f2df94b80011a8a93fa08f0046d1390cec49/examples/examples.ipynb">here</a>
  
<p><strong>Tabular</strong></p>

```python
from anonympy.pandas import dfAnonymizer
from anonympy.pandas.utils import load_dataset

df = load_dataset() 
print(df)
```

|   |  name | age |  birthdate |   salary |                                  web |                email |       ssn |
|--:|------:|----:|-----------:|---------:|-------------------------------------:|---------------------:|----------:|
| 0 | Bruce | 33  | 1915-04-17 | 59234.32 | http://www.alandrosenburgcpapc.co.uk | josefrazier@owen.com | 343554334 |
| 1 | Tony  | 48  | 1970-05-29 | 49324.53 | http://www.capgeminiamerica.co.uk    | eryan@lewis.com      | 656564664 |
  
```python
# Calling the generic Function
anonym = dfAnonymizer(df)
anonym.anonymize(inplace = False) # changes will be returned, not applied
```

|      | name            | age    | birthdate  | age     | web        |         email       |     ssn     |
|------|-----------------|--------|------------|---------|------------|---------------------|-------------|
| 0    | Stephanie Patel | 30     | 1915-05-10 | 60000.0 | 5968b7880f | pjordan@example.com | 391-77-9210 |
| 1    | Daniel Matthews | 50     | 1971-01-21 | 50000.0 | 2ae31d40d4 | tparks@example.org  | 872-80-9114 |
  
```python
# Or applying a specific anonymization technique to a column
from anonympy.pandas.utils import available_methods

anonym.categorical_columns
... ['name', 'web', 'email', 'ssn']
available_methods('categorical') 
... categorical_fake	categorical_fake_auto	categorical_resampling	categorical_tokenization	categorical_email_masking
  
anonym.anonymize({'name': 'categorical_fake', 
                  'age': 'numeric_noise',
                  'birthdate': 'datetime_noise',
                  'salary': 'numeric_rounding',
                  'web': 'categorical_tokenization', 
                  'email':'categorical_email_masking', 
                  'ssn': 'column_suppression'})
print(anonym.to_df())
```
|   |  name | age |  birthdate |   salary |                                  web |                email |
|--:|------:|----:|-----------:|---------:|-------------------------------------:|---------------------:|
| 0 | Paul Lang | 31  | 1915-04-17 | 60000.0 | 8ee92fb1bd | j*****r@owen.com |
| 1 | Michael Gillespie  | 42  | 1970-05-29 | 50000.0 | 51b615c92e    | e*****n@lewis.com      | 
 
<br >
<p><strong>Images</strong></p>

```python
# Passing an Image
import cv2
from anonympy.images import imAnonymizer

img = cv2.imread('sulking_boy.jpg')
anonym = imAnonymizer(img)

blurred = anonym.face_blur((31, 31), shape='r', box = 'r')  # blurring shape and bounding box ('r' / 'c')
cv2.imshow('Blurred', blurred)
```
`anonym.face_blur()`            |  `anonym.face_pixel()`            |    `anonym.face_SaP()`
:-------------------------:|:-------------------------:|:-------------------------:
![input_img1](https://raw.githubusercontent.com/ArtLabss/open-data-anonimizer/d61127f7a8fdff603af21dcab8edbf72f2aab292/examples/files/sad_boy_blurred.jpg)  |  ![output_img1](https://raw.githubusercontent.com/ArtLabss/open-data-anonimizer/d61127f7a8fdff603af21dcab8edbf72f2aab292/examples/files/sad_boy_pixel.jpg)    |   ![sap_image](https://raw.githubusercontent.com/ArtLabss/open-data-anonimizer/d61127f7a8fdff603af21dcab8edbf72f2aab292/examples/files/sad_boy_sap.jpg) 

```python
# Passing a Folder 
path = 'C:/Users/shakhansho.sabzaliev/Downloads/Data' # images are inside `Data` folder
dst = 'D:/' 
anonym = imAnonymizer(path, dst)

anonym.blur(method = 'median', kernel = 11) 
```

<p>This will create a folder <i>Output</i> in <code>dst</code> directory.</p>
<p>The <i>Data</i> folder had the following structure</p>

```
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
```

<p>The <i>Output</i> folder will have the same structure and file names but blurred images.</p>

<br>

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

<p><a href="https://github.com/ArtLabss/open-data-anonimizer/blob/main/LICENSE">BSD 3</a></p>


<h2>Code of Conduct</h2>
<p>Please see <a href="https://github.com/ArtLabss/open-data-anonimizer/blob/main/CODE_OF_CONDUCT.md">Code of Conduct</a>. 
All community members are expected to follow it.</p>
