import tensorflow as tf
import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import cv2
from tensorflow import keras
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D ,MaxPooling2D , Dropout , Flatten , BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping
import skimage

import glob

#########################전처리 시작 ###################################################
closed_img = [cv2.imread(file)[25:50] for file in glob.glob("C:/Users/ip111/dataset_B_FacialImages/ClosedFace/*.jpg")]
closed_img=np.array(closed_img)
open_img = [cv2.imread(file)[25:50] for file in glob.glob("C:/Users/ip111/dataset_B_FacialImages/OpenFace/*.jpg")]
open_img=np.array(open_img)
my_img=[cv2.resize(np.array(cv2.imread(file)),(100,100))[25:50] for file in glob.glob("C:/Users/ip111/myface-20211109T074142Z-001/myface/*.jpg")]
#my_img=[cv2.imread(file) for file in glob.glob("C:/Users/ip111/myface-20211109T074142Z-001/myface/*.jpg")]
my_img=np.array(my_img)
print(open_img.shape)
print(closed_img.shape)
print(my_img.shape)
print(my_img)
print(open_img)

for i in range(len(open_img)):
    open_img[i]= cv2.cvtColor(open_img[i], cv2.COLOR_BGR2RGB)
    
for i in range(len(closed_img)):
    closed_img[i]= cv2.cvtColor(closed_img[i], cv2.COLOR_BGR2RGB)               ###############for문 없이 하면 이미지의 디멘젼이 축소된다
    
for i in range(len(my_img)):
    my_img[i]= cv2.cvtColor(my_img[i], cv2.COLOR_BGR2RGB)
    
    


my_label=np.array([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])

open_img4train=open_img[:1100]
closed_img4train=closed_img[:1100]
open_img4test=open_img[1100:-39]
closed_img4test=closed_img[1100:] #######################학습과 검증셋 분류

open_label4train=np.zeros(shape=(len(open_img4train),), dtype=np.int8)
closed_label4train=np.ones(shape=(len(closed_img4train),), dtype=np.int8)
open_label4test=np.zeros(shape=(len(open_img4test),), dtype=np.int8)
closed_label4test=np.ones(shape=(len(closed_img4test),), dtype=np.int8)



train_images=np.concatenate((open_img4train,closed_img4train),axis=0)
train_labels=np.concatenate((open_label4train,closed_label4train),axis=0)
train_images=np.concatenate((train_images,my_img),axis=0)
train_labels=np.concatenate((train_labels,my_label),axis=0)


test_images=np.concatenate((open_img4test,closed_img4test),axis=0)
test_labels=np.concatenate((open_label4test,closed_label4test),axis=0)



plt.imshow(open_img[2])
plt.show()
plt.imshow(closed_img[2])
plt.show()
print(open_img.shape)



seed=0
np.random.seed(seed)
tf.random.set_seed(seed)

##########################섞기#######################################
train_idx=np.arange(len(train_images))
test_idx=np.arange(len(test_images))

np.random.shuffle(train_idx)
np.random.shuffle(test_idx)

train_images=train_images[train_idx]
train_labels=train_labels[train_idx]
test_images=test_images[test_idx]
test_labels=test_labels[test_idx]

##########################섞기########################################
#########################전처리 완료 ###################################################

###################학습#################################################################

sgd=keras.optimizers.SGD(lr=0.001, momentum=0.9, decay=0.0005 , nesterov=True)

#train_images = train_images.astype('float32')/255     
#test_images = test_images.astype('float32')/255     


train_labels=tf.keras.utils.to_categorical(train_labels)
test_labels=tf.keras.utils.to_categorical(test_labels) ####################원핫엔코딩을 한다. 

model = Sequential()

model.add(Conv2D(filters=96,kernel_size=(11,11),strides=(4,4),input_shape=(25,100,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
model.add(BatchNormalization())

model.add(Conv2D(256,(5,5),padding='same',activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
model.add(BatchNormalization())

model.add(Conv2D(384,(3,3),padding='same',activation='relu'))
model.add(Conv2D(384,(3,3),padding='same',activation='relu'))
model.add(Conv2D(256,(3,3),padding='same',activation='relu'))
#model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
model.add(BatchNormalization())

model.add(Flatten())
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(2,activation='softmax'))

model.compile(optimizer=sgd,loss='binary_crossentropy',metrics=['accuracy'])

model.summary()

MODEL_DIR='./model/'
if not os.path.exists(MODEL_DIR):
    os.mkdir(MODEL_DIR)
    

##모델저장조건설정
modelpath="./model/NSWD{epoch:02d}-{val_loss:.4f}.hdf5" # 이 이름으로 저장을 해라
checkpointer= ModelCheckpoint(filepath=modelpath,monitor='val_loss',verbose=1,save_best_only=True) # 여기에 저장하고! val_loss를 보고! 
# SAVE BEST ONLY ==> 전 값보다 지금 값이 더 작으면 저장해라. 

early_stopping_callback=EarlyStopping(monitor='val_loss',patience=10)

history=model.fit(train_images,train_labels,validation_data=(test_images,test_labels),epochs=20,batch_size=128,verbose=0,callbacks=[early_stopping_callback,checkpointer])


print("\n Test Accuracy: %.4f"%(model.evaluate(test_images,test_labels)[1]))
