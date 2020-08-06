from keras.models import Sequential
from keras.layers import Layer

from keras.layers.core import Dense,Activation,Dropout,Flatten,Layer
from keras.layers.convolutional import MaxPooling2D,Convolution2D
from keras.optimizers import SGD,Adam,RMSprop
from keras.utils import np_utils
from keras.layers.normalization import BatchNormalization
from array import *
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from tqdm import tqdm
from numpy import *
from PIL import Image
import theano
import sys
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import pandas as pd
import cv2
import glob
from sklearn.preprocessing import binarize
from sklearn.metrics import confusion_matrix

path1="C:\\dataset\\train"
path2="C:\\dataset\\test"

folders = glob.glob('C:/dataset/CleanImages/**')
imagenames_list = []
for folder in folders:
    for f in glob.glob(folder+'/*.png'):
        
        imagenames_list.append(f)
        
read_images = []        

for image in imagenames_list:
    
    read_images.append(cv2.imread(image, cv2.IMREAD_GRAYSCALE))
    




listing=os.listdir(path1)
num_samples=len(imagenames_list)
print(num_samples)

img_rows=50;
img_cols=50;
for file in listing:
  im=Image.open(path1 + '\\' + file)  
  img=im.resize((img_rows,img_cols))
  gray=img.convert('L') #need to do some more processing here          
  gray.save(path2 +'\\' +  file, "JPEG")
 
  imlist=os.listdir(path2)
#covert it into an array
im1 = array(Image.open(path2 + '\\'+ imlist[0]))
m,n=im1.shape[0:2]
imnbr=len(imlist)   
print(imnbr)
 

# immatrix=array([array(Image.open(path2+'\\'+im2)).flatten() for im2 in imlist],'f');
# print(immatrix.shape)

                

immatrix = array( [array(cv2.imread(imagenames_list[i], cv2.IMREAD_GRAYSCALE)).flatten() for i in range(len(imagenames_list))] ,'f')
print (immatrix.shape)
# img66=immatrix[10005].reshape(img_rows,img_cols)
# plt.imshow(img66)
# plt.imshow(img66,cmap='gray')



# categories=[]
# counts=[]
# rand_strs=[]
# str=[]

# for img_filename in os.listdir(path2):
    
#         id,count,id2,category= img_filename.split('.')[0].split('_')
#         str.append(id2)
#         categories.append(category)
#         counts.append(int(count))
#         rand_strs.append(id)
       
label =np.ones((num_samples,),dtype=int)

label[0:479]=1
label[480:959]=10
label[960:1439]=11
label[1440:1919]=12
label[1920:2399]=13
label[2400:2879]=14
label[2880:3359]=15
label[3360:3839]=16
label[3840:4319]=17
label[4320:4799]=18
label[4800:5279]=19
label[5280:5759]=2
label[5760:6239]=20
label[6240:6719]=21
label[6720:7199]=22
label[7200:7679]=23
label[7680:8159]=24
label[8160:8639]=25
label[8640:9119]=26
label[9120:9599]=27
label[9600:10079]=28
label[10080:10559]=3
label[10560:11039]=4
label[11040:11519]=5
label[11520:11999]=6
label[12000:12479]=7
label[12480:12959]=8
label[12960:]=9

print(num_samples)


data,label=shuffle(immatrix,label,random_state=2)
train_data=[data,label]
img=immatrix[13438].reshape(img_rows,img_cols)
plt.imshow(img)

plt.imshow(img,cmap='gray')


batch_size=32
nb_classes=29
nb_epoch=20
img_channels=1
nb_filters=32
nb_pool=2
nb_conv=3
x, y = np.arange(10).reshape((5, 2)), range(5)
(x,y)=(train_data[0],train_data[1])


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
x_train=x_train.astype('float32')
x_test=x_test.astype('float32')
x_train/=255
x_test/=255
print('X_train shape: ',x_train.shape)
print(x_train.shape[0],'train samples')
print(x_test.shape[0],'test samples')

y_train=np_utils.to_categorical(y_train,nb_classes)
y_test=np_utils.to_categorical(y_test,nb_classes)
i=100
print(y_train.shape)

#plt.imshow(x_train[i,0],interpolation='nearest')
print("label: ",y_train[i,:])



model = Sequential()

model.add(Convolution2D(nb_filters, nb_conv, nb_conv,activation="relu",
                        border_mode='valid',
                        input_shape=(img_rows, img_cols,1)))
convout1 = Activation('relu')
model.add(convout1)
model.add(Convolution2D(nb_filters, nb_conv, nb_conv))
convout2 = Activation('relu')
model.add(convout2)
model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
model.add(Dropout(0.5))


model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

x_train = x_train.reshape(-1,50, 50, 1)   #Reshape for CNN 
x_test = x_test.reshape(-1,50, 50, 1)

# model.compile(optimizer='rmsprop',
#               loss='categorical_crossentropy',
#               metrics=['accuracy'])

model.compile(optimizer='adadelta',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


model.fit(x_train, y_train, batch_size=batch_size,
          epochs=nb_epoch, verbose=1, validation_split=0.1)
# model.fit(x_train,y_train,verbose=1)
# # model.fit(x_train, y_train, batch_size=batch_size,
# #           epochs=nb_epoch, verbose=1, validation_data=(x_test,y_test))

score=model.evaluate(x_test,y_test,verbose=0)

print('Test score',score[0])
print('Test accuracy',score[1])
print(model.predict_classes(x_test[2000:2020]))
print(y_test[2000:2020])



# print(y_test)
# # y_pred_class = logreg.predict(x_test)
# y_pred_class=model.predict(x_test)
# print(y_pred_class)
# y_pred_class=np.argmax(y_test)
# cm=confusion_matrix(y_test,y_pred_class)
# print(cm)

y_pred_class=model.predict(x_test)
y_pred_class=np.argmax(y_pred_class, axis=1)
print(y_pred_class)
y_test=np.argmax(y_test, axis=1)
print(y_test)

print(metrics.accuracy_score(y_test, y_pred_class)) #percentage of correct predictions



y_test.mean()

1 - y_test.mean()

max(y_test.mean(), 1 - y_test.mean())
#confusion matrix
print(metrics.confusion_matrix(y_test, y_pred_class))

confusion = metrics.confusion_matrix(y_test, y_pred_class)
print(confusion.shape)
#[row, column]
TP = confusion[1, 1]
TN = confusion[0, 0]
FP = confusion[0, 1]
FN = confusion[1, 0]

print((TP + TN) / float(TP + TN + FP + FN))
print(metrics.accuracy_score(y_test, y_pred_class))



classification_error = (FP + FN) / float(TP + TN + FP + FN)

print(classification_error)
print(1 - metrics.accuracy_score(y_test, y_pred_class))


sensitivity = TP / float(FN + TP)

print(sensitivity)
print(metrics.recall_score(y_test, y_pred_class,pos_label='positive',average='micro'))

specificity = TN / (TN + FP)

print(specificity)

false_positive_rate = FP / float(TN + FP)

print(false_positive_rate)
print(1 - specificity)

precision = TP / float(TP + FP)
print(precision)
print(metrics.precision_score(y_test, y_pred_class,pos_label='positive',average='micro'))


y_pred_prob = model.predict_proba(x_test)[:, 1]
model.predict_proba(x_test)[0:10, 1]

plt.rcParams['font.size'] = 12
plt.hist(y_pred_prob, bins=8)

# x-axis limit from 0 to 1
plt.xlim(0,1)
plt.title('Histogram of predicted probabilities')
plt.xlabel('Predicted probability of diabetes')
plt.ylabel('Frequency')

y_pred_prob[0:1000]

print(confusion)

print(metrics.confusion_matrix(y_test, y_pred_class))


print (46 / float(46 + 16))
print(80 / float(80 + 50))



    
