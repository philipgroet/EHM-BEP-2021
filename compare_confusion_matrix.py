from PIL import Image
from io import BytesIO
from IPython.display import display
# from bs4 import BeautifulSoup
from urllib.parse import urlparse
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
TEST_DATA = BASE + '/grande'
TEST_BASE = BASE + '/grande'

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


import csv
import sys

y_pred_grande = []
y_true_grand = []

db_fileName = TEST_BASE + '/data-keypoints.csv'
db_file = open(db_fileName, 'r')
reader = csv.DictReader(db_file)
db_fileName2 = TEST_BASE + '/adjust.csv'
db_file2 = open(db_fileName2, 'r')
reader2 = csv.DictReader(db_file2)
sift = cv.SIFT_create()
csv.field_size_limit(sys.maxsize)

for images in reader2:
    #keyp_1 = []
    #keyp_2 = []
    desc_1 = []
    desc_2 = []
    
    print('Start search data:', images['img1'], ' &', images['img2'])
    db_file.seek(0)
    for data in reader:
        if images['img1'] == data['url']:
            #keyp_1 = deserialize_keypoints(data['keypoints'])
            desc_1 = data
        elif images['img2'] == data['url']:
            #keyp_2 = deserialize_keypoints(data['keypoints'])
            desc_2 = data
    
    # if desc_1 is None or desc_2 is None:
    #     print('Not in current dataset')
    #     continue
    
   
    bf = cv.BFMatcher()
    try:
        matches = bf.knnMatch(deserialize_descriptors(desc_1['descriptors']), deserialize_descriptors(desc_2['descriptors']), k=2) 
    except:
        print('Deze was dus te groot, of niet in de set')
        continue
    good = []
    # match_i = []

    for m,n in matches:
        # dist1 = m.distance
        # dist2 = n.distance
        # rela = dist1/dist2
        # match_i.append(rela)
        if m.distance < 0.6132918588356167*n.distance:
            good.append([m])

    # sorteer = np.sort(match_i)
    # perfPrint('filter matches')
    # print(type(match_i))
    # print(match_i)
    # vaag = np.array(match_i,dtype=float)
    # writer = csv.DictWriter(db_file, fieldnames=fieldnames)
    # writer.writerow({
    #     'url_needle' : images['img1'],  
    #     'url_haystack': images['img1'], 
    #     'keypoints' : len(matches), 
    #     'relatie': serialize_descriptors(sorteer[:300])
    # })

    perc = len(good)/len(matches)

    if perc >= 0.008: 
        print('Found a match!')
        # print(row['url'], ' and', row2['url'])
        print('Number of keypoints:', len(matches))
        print('Number of qualitive matches', len(good))
        print('Percentage of good matches: ', perc*100, '%')
        y_pred_grande.append(1) 
    else:
        y_pred_grande.append(0) 

    if images['match'] == '1':
        y_true_grand.append(1)
    else:
        y_true_grand.append(0)
    print('lengths:', len(y_pred_grande), ' &', len(y_true_grand))



from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# db_fileName = TEST_DATA + '/results_orb_notated.csv'
# db_file = open(db_fileName, 'r')
# reader = csv.DictReader(db_file)

def calculate_score(y_true, y_pred):
  
  matrix = metrics.confusion_matrix(y_true, y_pred)
  acc_score = metrics.accuracy_score(y_true, y_pred)

  #Results

  plt.figure(figsize=(4,4))
  sns.heatmap(matrix, annot=True, fmt=".0f", linewidths=.5, square = True, cmap = 'Blues');
  plt.ylabel('Actual label');
  plt.xlabel('Predicted label');
  all_sample_title = 'Accuracy Score: {:.3f}'.format(acc_score)
  plt.title(all_sample_title, size = 10);


calculate_score(y_true_grand, y_pred_grande)