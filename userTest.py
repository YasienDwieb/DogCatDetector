import pickle
import cv2
import os
from sklearn.svm import SVC
import numpy as np

train_desc=list()
with open ('Training 2000-40/train_desc.txt', 'rb') as fp:
     train_desc = pickle.load(fp)

train_labels=list()
with open ('Training 2000-40/train_labels.txt', 'rb') as fp:
     train_labels = pickle.load(fp)

dictionary =np.load('Training 2000-40/dictionary.npy')

sift2 = cv2.xfeatures2d.SIFT_create()
bowDiction = cv2.BOWImgDescriptorExtractor(sift2, cv2.BFMatcher(cv2.NORM_L2))
bowDiction.setVocabulary(dictionary)

def feature_extract(pth):
    im = cv2.imread(pth, 0)
    return bowDiction.compute(im, sift2.detect(im))

clf = SVC()    
clf.fit(train_desc,train_labels)    

print 'Images should be placed in neighbor folder called testing'
print 'Or an error would occur'

while(True):
  testImage=raw_input('Enter photo name (e.g:1000.jpg): ')
  
  if os.path.isfile('customTesting/'+testImage):
    features=feature_extract('customTesting/'+testImage)
    result=clf.predict(features)
    if(result==1):
      print 'This is a Dog'
    elif(result==2):
      print 'This is a Cat'
    else:
      print 'This is not a defined animal'
    
    another=raw_input('Try another Image?(y/n)')
    if another not in ('yes', 'Yes', 'y', 'Y'):
      break
  else:
    print 'This is not a valid test image'
