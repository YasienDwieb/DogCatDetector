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

cats="testing/cats/"
dogs="testing/dogs/"


cats_results={}
for p in os.listdir(cats):
  features=feature_extract(cats+p)
  cats_results[p]=clf.predict(features)
	
dogs_results={}
for p in os.listdir(dogs):
  features=feature_extract(dogs+p)
  dogs_results[p]=clf.predict(features)


dogs_count=sum( x == 1 for x in dogs_results.values() )
cats_count=sum( x == 2 for x in cats_results.values() )
    
print 'dogs percntage is ',dogs_count[0]*1.0/len(dogs_results)
print 'cats percntage is ',cats_count[0]*1.0/len(cats_results)
