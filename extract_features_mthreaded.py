from PIL import Image
from io import BytesIO
from IPython.display import display
# from bs4 import BeautifulSoup
from urllib.parse import urlparse
from numpy.lib.utils import source
#from google.colab import drive
import requests
import os
import math
import glob
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import csv


def getExt(urls):
    exts = []
    for url in urls:
        path = urlparse(url).path
        ext = os.path.splitext(path)[1]
        exts.append(ext)
    return exts

#Filter out items in a dataset that are not images
def filterImages(paths):
    allowed_paths = []
    for path in paths:
        ext = os.path.splitext(path)[::-1]
        #print("ext: ", ext)
        allowedExt = [".png", ".jpg", ".jpeg"]
        if ext[0].lower() in allowedExt:
            allowed_paths.append(path)
    return allowed_paths


# def getTargetImageUrls(folderUrl):
#   targetExts = ['.jpg','.png']
#   page = requests.get(folderUrl).text
#   soup = BeautifulSoup(page, 'html.parser')
#   images = [folderUrl + '/' + node.get('href') for node in soup.find_all('a') if getExt(node.get('href')).lower() in targetExts]
#   return images

def getTargetImageUrls():
  return glob.glob(os.path.join(TARGET_IMAGES_FOLDER,'*'))
  
def getImageFromUrl(url):
    isAbsolute = True if '//' in url else False
    try:
        # image = Image.open(BytesIO(requests.get(url).content)) if isAbsolute else Image.open(url)
        if isAbsolute:
            nparr = np.fromstring(requests.get(url).content, np.uint8)
            image = cv.imdecode(nparr, cv.IMREAD_COLOR)
        #image = cv.imread(BytesIO(requests.get(url).content))
        else: 
            image = cv.imread(url)
    except:
        image = None
    return image

def getImageUrls(folder):
    return glob.glob(os.path.join(folder,'*.jpg')) 

def getVideoFromUrl(url):
    pass

# TestImageUrls = getImageUrls(EIFFEL_HAYSTACK)
# TestExts = getExt(TestImageUrls)
# TestFilteredImages = filterImages(TestExts)




BASE = "."

WIKI_DATA = BASE + '/Datasets/Wikipedia_images'

TARGET_IMAGES_FOLDER = BASE + "/Search Engines for Digital History/Target images (Project 1 and 2)"
IMAGE_FILES_FOLDER_PROJECT_1 = BASE + "/Search Engines for Digital History/Project 1"
VIDEO_FILES_FOLDER_PROJECT_2 = BASE + "/Search Engines for Digital History/Project 2"
VIDEO_FILES_FOLDER_PROJECT_3 = BASE + "/Search Engines for Digital History/Project 3"

EIFFEL_NEEDLE = BASE + '/Datasets/Eiffel/Needle'
EIFFEL_HAYSTACK = BASE + '/Datasets/Eiffel/Haystack'
EIFFEL_HAYSTACK_SMALL = BASE + '/Datasets/Eiffel/Haystack small'
EIFFEL_HAYSTACK_ROT = BASE + '/Datasets/Eiffel/Haystack rotation'
EIFFEL_NEEDLE_SINGLE = BASE + '/Datasets/Eiffel/Needle small'
EIFFEL_HAYSTACK_SINGLE = BASE + '/Datasets/Eiffel/Haystack small'

ART_NEEDLE = BASE + '/Datasets/Art/Needle'
ART_HAYSTACK = BASE + '/Datasets/Art/Haystack'
ART_NEEDLE_WIKI = BASE + '/Datasets/Art/NEEDLE_WIKI'

HARRY_data = BASE + '/Datasets/HARRYPOTTER'

MESSI_NEEDLE = BASE + '/Datasets/Messi/Needle messi'
MESSI_HAYSTACK = BASE + '/Datasets/Messi/Haystack messi'

TEST_NEEDLE = BASE + '/Datasets/test_data/needle'
TEST_HAYST = BASE + '/Datasets/test_data/haystack'
TEST_DATA = BASE + '/ehm_dataset'
TEST_BASE = BASE + '/ehm_dataset'

#drive.mount('/content/drive', force_remount=True)



# Serialization functies

import pickle
import cv2 as cv
import codecs

def write_progress(nextName):
    f = open("wikipedia-next-name.txt", "w")
    f.write(nextName)
    f.close()
def read_progress():
    pass
    try:
        f = open("wikipedia-next-name.txt", "r")
        resumeName = f.read()
        print('Starting loading at ' + resumeName)
        return resumeName
    except:
        print("Could not load progress, beginning at the start")
        return ''

def serialize_descriptors(descr):
    return codecs.encode(pickle.dumps(descr), "base64").decode()
  # return pickle.dumps(descr, protocol=0) # Protocol=0 is printable ascii
def deserialize_descriptors(ser):
    return pickle.loads(codecs.decode(ser.encode(), "base64"))

def serialize_keypoints(keyps):
    simplified = []

    for keyp in keyps:
        simplified.append((
        keyp.pt, 
        keyp.size, 
        keyp.angle, 
        keyp.response, 
        keyp.octave, 
        keyp.class_id
        ))

    return pickle.dumps(simplified, protocol=0)

def deserialize_keypoints(simplified):
    keypoints = []

    unpickled = pickle.loads(simplified)

    for simp in unpickled:
        keypoint = cv.KeyPoint(x=simp[0][0],y=simp[0][1],_size=simp[1], _angle=simp[2], _response=simp[3], _octave=simp[4], _class_id=simp[5])
        
        keypoints.append(keypoint)

    return keypoints

def write_progress(nextName):
    f = open("wikipedia-next-name.txt", "w")
    f.write(nextName)
    f.close()
def read_progress():
    pass
    try:
        f = open("wikipedia-next-name.txt", "r")
        resumeName = f.read()
        print('Starting loading at ' + resumeName)
        f.close()
        return resumeName
    except:
        print("Could not load progress, beginning at the start")
        return ''




import cv2 as cv

# Scales an image down to fit in the bounding maxWidth and maxHeight 
# Image should be OpenCV image object
def boundFit(img, maxWidth, maxHeight):
    # If already fitting in side max
    if (img.shape[0] < maxHeight and img.shape[1] < maxWidth):
        return img

    # If height smaller than width
    if (img.shape[0] < img.shape[1]):
        scalePercent = maxWidth/img.shape[1]*100
    else:
        scalePercent = maxHeight/img.shape[0]*100

    newWidth = int(img.shape[1] * scalePercent / 100)
    newHeight = int(img.shape[0] * scalePercent / 100)

    newSize = (newWidth, newHeight)
    return cv.resize(img, newSize, interpolation = cv.INTER_AREA)





import numpy as np
import csv
import cv2 as cv
import threading
import time
import concurrent.futures
import logging
logging.root.setLevel(logging.DEBUG)

class String(object):
    def __init__(self, string):
        self.string = string

    def __sub__(self, other):
        if self.string.startswith(other.string):
            return self.string[len(other.string):]

    def __str__(self):
        return self.string

sift = cv.SIFT_create()

db_fileName = TEST_BASE + '/data-keypoints.csv'
print("Creating index at path: ", db_fileName)
db_file = open(db_fileName, 'a', newline='')
fieldnames = ['url', 'extractTime', 'keypoints', 'descriptors']
writer = csv.DictWriter(db_file, fieldnames=fieldnames)
writer.writeheader()


write_lock = threading.Lock()
def write_row(data):
  logging.debug('Waiting for csv lock')
  with write_lock:
    logging.debug('Writing csv')
    writer.writerow(data)
    logging.debug('csv write done')

def thread_func(root, file):
    ext = os.path.splitext(file)[::-1]
    #print("ext: ", ext)
    allowedExt = [".png", ".jpg", ".jpeg"]
    if ext[0].lower() not in allowedExt:
        # print('Extension at', os.path.join(root, file), 'not allowed, ', ext[0].lower())
        return

    sourceImage = cv.imread(os.path.join(root, file))

    # Convert from 8u_int to 8u
    logging.debug(os.path.join(root, file))

    startTime = time.time()

    # sourceImage = boundFit(sourceImage, 3000, 3000)
    sourceImage = cv.normalize(sourceImage, None, 0, 255, cv.NORM_MINMAX).astype('uint8')
    try:
        keypoints_grande_data, descriptors_grande_data = sift.detectAndCompute(sourceImage, None)
    except Exception as e:
        logging.debug('Detect and compute failed: ')
        logging.debug(e)
        return

    write_row({
        'url': String(os.path.join(root, file)) - String(TEST_BASE + '/'), 
        'extractTime': time.time()-startTime,
        'keypoints': serialize_keypoints(keypoints_grande_data), 
        'descriptors': serialize_descriptors(descriptors_grande_data)
    })


with concurrent.futures.ThreadPoolExecutor(max_workers=9) as executor:
    for root, subdirectories, files in os.walk(TEST_BASE):
        for file in files:
            executor.submit(thread_func, root, file)
        
    

print("Successfully indexed images")
db_file.close()