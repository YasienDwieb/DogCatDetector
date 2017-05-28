import cv2
import numpy as np
import os
from sklearn.svm import SVC
import pickle


path = "training"
training_names = os.listdir(path)

training_paths = []
names_path = []
#get full list of all training images
for p in training_names:
    training_paths1 = os.listdir(path+"/"+p)
    for j in training_paths1:
        training_paths.append(path+'/'+p+'/'+j)
        names_path.append(p)

sift = cv2.xfeatures2d.SIFT_create()

dictionarySize = 40

BOW = cv2.BOWKMeansTrainer(dictionarySize)

print 'Extracting keypoints and descriptors'

for p in training_paths:
    image = cv2.imread(p,0)
    kp, dsc= sift.detectAndCompute(image, None)
    BOW.add(dsc)

#dictionary created
dictionary = BOW.cluster()

sift2 = cv2.xfeatures2d.SIFT_create()
bowDiction = cv2.BOWImgDescriptorExtractor(sift2, cv2.BFMatcher(cv2.NORM_L2))
bowDiction.setVocabulary(dictionary)

def feature_extract(pth):
    im = cv2.imread(pth, 0)
    return bowDiction.compute(im, sift.detect(im))


print 'BOW construction'

train_desc = []
train_labels = []
i = 0
for p in training_paths:
    train_desc.extend(feature_extract(p))
    if names_path[i]=='dogs':
        train_labels.append(1)
    if names_path[i]=='cats':
        train_labels.append(2)
    i = i+1
    
clf = SVC()    
clf.fit(train_desc,train_labels)    

with open('train_desc.txt', 'wb') as fp:
    pickle.dump(train_desc, fp)

with open('train_labels.txt', 'wb') as fp:
    pickle.dump(train_labels, fp)

np.save('dictionary.npy',dictionary)

print 'Testing input images'

test_results={}
for p in os.listdir('testing'):
  features=feature_extract('testing/'+p)
  test_results[p]=clf.predict(features)
	
print 'Done making the model'
print 'You will find it at ./train_desc.txt , ./train_labels.txt ./dictionary.npy'
