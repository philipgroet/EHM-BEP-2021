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






# import csv
# import sys


# def threshold_performance(th):
#     rela_good = []
#     rela_worse = []
#     not_found = 0
#     groter_dan_5 = 0

#     db_fileName = TEST_BASE + '/threshold.csv'
#     db_file = open(db_fileName, 'r')
#     reader = csv.DictReader(db_file)

#     for row in reader:
        
#         print('.', end='')
#         matches_need = math.ceil(th*int(row['keypoints'])) # Number of matches needed (th percent of total keypoints should be match)
#         if matches_need >= 500: # Limit
#           groter_dan_5 = groter_dan_5 + 1
#           matches_need = 499
#           # continue
#         # elif matches_need < 20:
#         #   continue

#         relation = deserialize_descriptors(row['relatie'])
#         # print('len rel', len(relation), )

#         # try:
#         if row['match'] == '1':
#             rela_good.append(relation[matches_need])
#         else:
#             rela_worse.append(relation[matches_need])
#         # except Exception as e:
#         #   print('\t', matches_need)
#         #   print('Problem: ', e)

#     db_file.close()
        
#     rela_good = np.sort(rela_good)
#     rela_worse = np.sort(rela_worse)
#     max_th = max(rela_good)
#     min_th = min(rela_worse)
    

#     if max_th < min_th:
#         print(th, ' is a good threshold')
#     else:
#         for i in rela_good:
#             if i > min_th:
#                 not_found = not_found + 1
#         #print('Not found matches:' not_found, ' using')
#     perform = 1 - (not_found/len(rela_good))

#     #print('Result of threshold', th, ' is', perform*100, '%')

#     return perform*100, not_found, min_th, groter_dan_5

# best_th = 0
# best_thresh = 0
# best_perf = 0
# best_gd5 = 0

# for i in range (1,39):
#     th = i * 0.001
#     perf, not_found, thresh, gd5 = threshold_performance(th)
#     # print('Result of threshold', th, ' is', perf, '%')
#     # print('This means that', not_found, ' matches are missed')
#     # print('The match threshold for this case shoud be:', thresh)
#     if best_perf < perf:
#         best_th = th
#         best_thresh = thresh
#         best_perf = perf
#         best_gd5 = gd5
# print('The optimal threshold is', best_th, ' using match threshold', best_thresh)
# print('Which results in a performance of', best_perf, ' %')
# print('Groter dan 500 matches nodig geskipt:', best_gd5)
    


import csv
import sys
import cv2 as cv
import numpy as np

#resetPerf()

def threshold_performance(th):
    
    
    balance_opt = [0, 0, 0, 0, 0]
    precision_opt = [0, 0, 0, 0, 0]
    recall_opt = [0, 0, 0, 0, 0]
    db_fileName = TEST_BASE + '/threshold.csv'
    #db_fileName = BASE + '/Datasets/test_data/data_dist_labels.csv'
    db_file = open(db_fileName, 'r')
    reader = csv.DictReader(db_file)

    for min_match in range (1,30):
        db_file.seek(0)
        rela_good = np.array([])
        rela_worse = np.array([])

        #perfPrint(None)
        for row in reader:
            try:
                matches_need = math.ceil(th*int(row['keypoints']))
            except:
                continue
            
            
            if matches_need >= 500:
                continue
            
            if matches_need < min_match:
                matches_need = min_match

            
            relation = deserialize_descriptors(row['relatie'])
            #print(matches_need)
            if row['match'] == '1':
                rela_good = np.append(rela_good, relation[matches_need])
            else:
                rela_worse = np.append(rela_worse, relation[matches_need])

        #perfPrint('Done reading threshold_file')
                
        rela_good = np.sort(rela_good)
        rela_worse = np.sort(rela_worse)
        
        max_th = max(rela_good)
        min_th = min(rela_worse)
        #print(rela_worse)
        #print(min_match, th_point)
            # (((TP/(TP+FN)+(TN/(TN+FP))) / 2
            # TP/TP+FP
    
        if max_th < min_th:
            print(th, ' is a good threshold')
            opt_th = min_th
        else:
            for dec_th in rela_worse:
                #perfPrint(None)
                FN = 0
                FP = 0
                TN = 0
                TP = 0


                # for i in rela_good:
                #     if i > dec_th:
                #         FN = FN + 1
                # for j in rela_worse:
                #     if j < dec_th:
                #         FP = FP + 1
                
                FN = np.count_nonzero(np.where(rela_good > dec_th))
                FP = np.count_nonzero(np.where(rela_worse < dec_th))

                TP = len(rela_good) - FN
                TN = len(rela_worse) - FP
                # print(TN, FN)
                # print(FP, TP)
                balanced_acc = 0.5*((TP/(TP+FN))+(TN/(TN+FP)))
                try:
                    precision = TP/(TP+FP)
                    recall = TP/(TP+FN)
                except:
                    precision = 0
                    recall = 0
                # print(balanced_acc)
                # print(best_balance)
                # print('The optimal threshold ', precision, th)
                # print('The optimal threshold ', recall, min_match)
                # print('The optimal threshold ', balanced_acc)
                if precision > precision_opt[4]:
                    precision_opt = [balanced_acc, dec_th, min_match, recall, precision, th, TN, TP, FN, FP]
                elif precision == precision_opt[4] and recall >= precision_opt[3] and balanced_acc >= precision_opt[0]:
                    precision_opt = [balanced_acc, dec_th, min_match, recall, precision, th, TN, TP, FN, FP]

                if balanced_acc > balance_opt[0]:
                    balance_opt = [balanced_acc, dec_th, min_match, recall, precision, th, TN, TP, FN, FP]
                elif balanced_acc == balance_opt[0] and recall >= balance_opt[3] and precision >= balance_opt[4]:
                    balance_opt = [balanced_acc, dec_th, min_match, recall, precision, th, TN, TP, FN, FP]

                if recall > recall_opt[3]:
                    recall_opt = [balanced_acc, dec_th, min_match, recall, precision, th, TN, TP, FN, FP]
                elif recall == recall_opt[3] and precision >= recall_opt[4] and balanced_acc >= recall_opt[0]:
                    recall_opt = [balanced_acc, dec_th, min_match, recall, precision, th, TN, TP, FN, FP]
                #perfPrint('for dec_th in rela_worse')

    

    return precision_opt, balance_opt, recall_opt



import pandas

# db_fileName = TEST_BASE + '/threshold.csv'
# thresholds_df = pandas.read_csv(db_fileName)
# thresholds_df['relatie'] = thresholds_df.apply(lambda row : deserialize_descriptors(row['relatie']))

threshFilePath = TEST_BASE + '/threshold.csv'
#db_fileName = BASE + '/Datasets/test_data/data_dist_labels.csv'
threshFile = open(threshFilePath, 'r')
reader = csv.DictReader(threshFile)

def threshold_performance_v2(keypoint_th, min_abs_matches, match_ratio_th):
    threshFile.seek(0)
    next(reader)


    rela_good = np.array([])
    rela_worse = np.array([])

    for row in reader:
        # try:
        matches_need = math.ceil(keypoint_th*int(row['keypoints']))
        # except:
        #     continue
        
        # We only save the best 500 matches in threshold.csv
        if matches_need >= 500:
            matches_need = 499
            continue
        
        # Minstens min_abs_matches nodig om een match te zijn, voorkomt enkele matches die meer dan 1% worden
        if matches_need < min_abs_matches:
            matches_need = min_abs_matches

        
        relation = deserialize_descriptors(row['relatie'])
        #print(matches_need)
        if row['match'] == '1':
            rela_good = np.append(rela_good, relation[matches_need])
        else:
            try:
                rela_worse = np.append(rela_worse, relation[matches_need])
            except:
                print('rela_worse param: ', rela_worse, matches_need, relation)
                exit()



    rela_good = np.sort(rela_good)
    rela_worse = np.sort(rela_worse)
    
    # max_th = max(rela_good)
    # min_th = min(rela_worse)

    FN = 0
    FP = 0
    TN = 0
    TP = 0

    # for i in rela_good:
    #     if i > dec_th:
    #         FN = FN + 1
    # for j in rela_worse:
    #     if j < dec_th:
    #         FP = FP + 1
    
    FN = np.count_nonzero(np.where(rela_good > match_ratio_th))
    FP = np.count_nonzero(np.where(rela_worse < match_ratio_th))

    TP = len(rela_good) - FN
    TN = len(rela_worse) - FP
    # print(TN, FN)
    # print(FP, TP)
    balanced_acc = 0.5*((TP/(TP+FN))+(TN/(TN+FP)))
    try:
        precision = TP/(TP+FP)
        recall = TP/(TP+FN)
    except:
        precision = 0
        recall = 0

    return precision, recall, balanced_acc, FN, FP, TN, TP



fieldnames = ['keypoint_th', 'min_abs_match', 'match_ratio_th', 'precision', 'recall', 'balanced', 'FN', 'FP', 'TN', 'TP']

optimal_keypoint_th = 0.037
optimal_min_abs_match = 21
optimal_match_ratio_th = 0.5519978894151195

precision, recall, balanced, FN, FP, TN, TP = threshold_performance_v2(optimal_keypoint_th, optimal_min_abs_match, optimal_match_ratio_th)
print('Performance best: p:', precision, 'r:', recall, 'b:', balanced, 'FN:', FN, 'FP', FP, 'TN', TN, 'TP', TP)


# Vary keypoint thresh
db_fileName = TEST_BASE + '/PRC_keypoint_th.csv'
db_file = open(db_fileName, 'a', newline='')
writer = csv.DictWriter(db_file, fieldnames=fieldnames)
writer.writeheader()

for keypoint_th in np.linspace(0, 0.15, 200):
    precision, recall, balanced, FN, FP, TN, TP = threshold_performance_v2(keypoint_th, optimal_min_abs_match, optimal_match_ratio_th)
    
    print('Param for: ', keypoint_th, optimal_min_abs_match, optimal_match_ratio_th, '\n\tperf: ', precision, recall, balanced)

    writer.writerow({
        'keypoint_th': keypoint_th, 
        'min_abs_match': optimal_min_abs_match, 
        'match_ratio_th': optimal_match_ratio_th,
        'precision': precision,
        'recall': recall,
        'balanced': balanced,
        'FN': FN,
        'FP': FP,
        'TN': TN,
        'TP': TP
    })

db_file.close()
print('Done!')
 


# Vary min_abs_match
db_fileName = TEST_BASE + '/PRC_min_abs_match.csv'
db_file = open(db_fileName, 'a', newline='')
writer = csv.DictWriter(db_file, fieldnames=fieldnames)
writer.writeheader()

for min_abs_match in range(0, 80):
    precision, recall, balanced, FN, FP, TN, TP = threshold_performance_v2(optimal_keypoint_th, min_abs_match, optimal_match_ratio_th)
    
    print('Param for: ', optimal_keypoint_th, min_abs_match, optimal_match_ratio_th, '\n\tperf: ', precision, recall, balanced)

    writer.writerow({
        'keypoint_th': optimal_keypoint_th, 
        'min_abs_match': min_abs_match, 
        'match_ratio_th': optimal_match_ratio_th,
        'precision': precision,
        'recall': recall,
        'balanced': balanced,
        'FN': FN,
        'FP': FP,
        'TN': TN,
        'TP': TP
    })

db_file.close()
print('Done!')



# Vary optimal_match_ratio_th
db_fileName = TEST_BASE + '/PRC_match_ratio_th.csv'
db_file = open(db_fileName, 'a', newline='')
writer = csv.DictWriter(db_file, fieldnames=fieldnames)
writer.writeheader()

for match_ratio_th in np.linspace(0, 1, 300):
    precision, recall, balanced, FN, FP, TN, TP = threshold_performance_v2(optimal_keypoint_th, optimal_min_abs_match, match_ratio_th)
    
    print('Param for: ', optimal_keypoint_th, optimal_min_abs_match, match_ratio_th, '\n\tperf: ', precision, recall, balanced)

    writer.writerow({
        'keypoint_th': optimal_keypoint_th, 
        'min_abs_match': optimal_min_abs_match, 
        'match_ratio_th': match_ratio_th,
        'precision': precision,
        'recall': recall,
        'balanced': balanced,
        'FN': FN,
        'FP': FP,
        'TN': TN,
        'TP': TP
    })

db_file.close()
print('Done!')


#print_avergae_runtimes()
    