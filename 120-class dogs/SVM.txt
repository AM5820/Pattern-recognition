#Knn
#upload dataset
#------------------------------------------------------------------------------------------------------------
import numpy as np
from PIL import Image
from PIL import ImageOps
import glob
import os
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn import svm


dir = 'working1'
ls=[]
labels = []
subdirs = [x[1] for x in os.walk(dir)]
k=0
for subdir in subdirs[0]:
    k+=1
    print(str(subdir))
    for filename in glob.glob('working1/'+str(subdir) +'/*.jpg'):
        image = Image.open(filename)
        size = (50, 50)
        image = ImageOps.fit(image, size, Image.ANTIALIAS)#.convert('LA')
        pix = np.array(image)
        pix = pix.flatten()
        pix = pix.tolist()
        ls.append(pix)
        labels.append(k)

#---------------------------------------------------------------------------------------------------------

dataset = np.array(ls)
labels = np.array(labels)
dataset.shape

X_train,X_test,y_train,y_test = train_test_split(dataset,labels,random_state=20)

#---------------------------------------------------------------------------------------------------------



clf = svm.SVC(C=3,gamma='scale')
clf.fit(X_train,y_train)
print(clf.score(X_test,y_test)*100)

----------------------------------------------------------------------------------------------------------