#Knn
#upload dataset
#------------------------------------------------------------------------------------------------------------
import numpy as np
from PIL import Image
from PIL import ImageOps
import glob
import os
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


import numpy as np
from PIL import Image
from PIL import ImageOps
import glob
import os
dir = 'fruits/Training'
ls=[]
labels = []
subdirs = [x[1] for x in os.walk(dir)]
k=0
for subdir in subdirs[0]:
    k+=1
    print(str(subdir))
    c=0
    for filename in glob.glob('fruits/Training/'+str(subdir) +'/*.jpg'):
        if(c!=100):
            image = Image.open(filename)
            size = (60, 60)
            image = ImageOps.fit(image, size, Image.ANTIALIAS)#.convert('LA')
            pix = np.array(image)
            pix = pix.flatten()
            pix = pix.tolist()
            ls.append(pix)
            labels.append(k)
            c+=1
        else:
            break

#---------------------------------------------------------------------------------------------------------

dataset = np.array(ls)
labels = np.array(labels)


X_train,X_test,y_train,y_test = train_test_split(dataset,labels,random_state=0)

#---------------------------------------------------------------------------------------------------------



knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train,y_train)
print("at k = "+str(1)+" , accuracy =  "+str(knn.score(X_test,y_test)*100)+str("%"))

#----------------------------------------------------------------------------------------------------------