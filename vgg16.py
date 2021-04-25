from contextlib import suppress
import matplotlib
import matplotlib.pyplot as plt

import numpy as np
import os
import shutil
from skimage.io import imread, imsave

from keras import applications
from keras import optimizers
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, Flatten
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import accuracy_score, classification_report


# In[2]:


plt.imshow(imread('C:/project/DataSets/Images/forest/forest00.tif'))
plt.show()


# In[3]:


plt.imshow(imread('C:/project/DataSets/Images/golfcourse/golfcourse00.tif'))
plt.show()

# In[4]:


plt.imshow(imread('C:/project/DataSets/Images/beach/beach00.tif'))
plt.show()


# In[5]:


plt.imshow(imread('C:/project/DataSets/Images/buildings/buildings00.tif'))
plt.show()



# ## The Various Classes of Land Use are listed

# In[6]:


# source_dir = os.path.join( 'Images')
# class_names = os.listdir(source_dir)
# class_names

labels=[
 'agricultural',
 'airplane',
 'baseballdiamond',
 'beach',
 'buildings',
 'chaparral',
 'denseresidential',
 'forest',
 'freeway',
 'golfcourse',
 'harbor',
 'intersection',
 'mediumresidential',
 'mobilehomepark',
 'overpass',
 'parkinglot',
 'river',
 'runway',
 'sparseresidential',
 'storagetanks',
 'tenniscourt']


# ## Creating Train, Test and Validation datasets

#  In the code section given below, 80 images are randomly selected from each class. The training dataset. Similarly, the remaining 10 images are selected and assigned to the validation dataset and 10 images are assigned to the testing dataset. All three datasets are mutually exclusive. that is, there is no common image among the three datasets

# In[7]:


def complement (a,b):
    f2=[]
    for x in a:
        x=os.path.splitext(x)[0]+'.tif'
        if x not in os.listdir(path1_train):
            f2.append(x)
    return (f2)


# In[8]:


# Selecting images for train
train_data_size=80
validate_data_size=10
test_data_size=10
for j in labels:
    path=os.path.join('DataSets/Images',j)
    path1_train=os.path.join('DataSets/Images/dataset','train',j)
    path1_validate=os.path.join('DataSets/Images/dataset','validate',j)
    path1_test=os.path.join('DataSets/Images/dataset','test',j)
    
    #shutil.rmtree(path1_train)
    #os.makedirs(path1_train) 
    #shutil.rmtree(path1_validate)
    #os.makedirs(path1_validate) 
    #shutil.rmtree(path1_test)
    #os.makedirs(path1_test) 
    
    
    files1= os.listdir(path)
    files1=files1[1:len(files1)]
    files1= np.random.permutation(files1)
    for i in range(0,train_data_size):
        file=os.path.join(path,files1[i])
        img1=imread(file)
        n=os.path.splitext(file)
        n=n[0].split('\\')
        n1=os.path.join(path1_train, n[2] +'.png')
        imsave(n1,img1)

    print(path1_train, len(os.listdir(path1_train)))
    validate_data0 = complement(files1, os.listdir(path1_train))
    
    validate_data = np.random.permutation(validate_data0)
    for i in range( 0, validate_data_size):
        file=os.path.join(path,validate_data[i])
        img1=imread(file)
        n=os.path.splitext(file)
        n=n[0].split('\\')
        n1=os.path.join(path1_validate, n[2] +'.png')
        imsave(n1,img1)
    print(path1_validate, len(os.listdir(path1_validate))) 
    
    test_data0 = complement(validate_data0, os.listdir(path1_validate))
    test_data = np.random.permutation(test_data0)
    for i in range(0, test_data_size):
        file=os.path.join(path,test_data[i])
        img1=imread(file)
        n=os.path.splitext(file)
        n=n[0].split('\\')
        n1=os.path.join(path1_test, n[2] +'.png')
        imsave(n1,img1)
        
    print(path1_test, len(os.listdir(path1_test)))


# The ImageDataGenerator is a function in Keras to convert images into arrays of a suitable size. This function has been used extensively in the code.

# In[9]:


base_directory=path=os.path.join('DataSets/Images/')
train_dir=os.path.join('C:/project/DataSets/Images/dataset/','train')
validation_dir=os.path.join('C:/project/DataSets/Images/dataset/','validate')
test_dir=os.path.join('C:/project/DataSets/Images/dataset/','test')


train_datagen= ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
train_dir,
target_size=(150, 150),
batch_size=32,
class_mode=None)




from keras.applications import VGG16
conv_base = VGG16(weights='imagenet',
include_top=False)
conv_base.summary()




def extract_features(directory, sample_count,batch_size):
    datagen = ImageDataGenerator(rescale=1./255)
    generator = datagen.flow_from_directory(
    directory,
    batch_size=batch_size,
    shuffle=False)
    class_names=generator.class_indices
    
    image_count = 0
    X_batches, Y_batches = [], []
    for inputs_batch, labels_batch in generator:
        X_batches.append(conv_base.predict(inputs_batch))
        Y_batches.append(labels_batch)
        image_count += inputs_batch.shape[0]
        # Must interrupt image_generator
        if image_count >= generator.n:
                break
        X = np.concatenate(X_batches)  
        Y = np.concatenate(Y_batches)

    return X, Y,class_names


# In[12]:



train_features, train_labels,class_names1 = extract_features(train_dir, 1681,64)


# In[13]:


validation_features, validation_labels,c_names1 = extract_features(validation_dir, 300,10)
test_features, test_labels,c_names2 = extract_features(test_dir, 300,10)


# In[14]:


labels=os.listdir(train_dir)

train_features.shape


# In[15]:


len(os.listdir(validation_dir))
len(labels)


# ## Model Implemented on Validation set

# A fully connected model is defined. and tested on the validation set.

# In[16]:


from keras import models
from keras import layers
from keras import optimizers
def model1(shape,n_labels):
    model = Sequential()
    model.add(Flatten(input_shape=shape))
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(n_labels, activation='softmax'))
    return model


model=model1(train_features.shape[1:],len(labels))
model.compile(optimizer=optimizers.adam(lr=2e-5),
loss='categorical_crossentropy',
metrics=['acc'])
history = model.fit(train_features, train_labels,
epochs=10,
batch_size=64,
validation_data=(validation_features, validation_labels))
model.save('VGG16_model.h5')


# In[17]:


import matplotlib.pyplot as plt
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()


# In[18]:


print(validation_labels.shape)
print(train_labels.shape)


# In[19]:


print(model.summary())


# ##  Model Implemented on Testing Set 

# In[20]:


import cv2
#fit on testing data
model=model1(train_features.shape[1:],len(labels))
model.compile(optimizer=optimizers.adam(lr=2e-5),loss='categorical_crossentropy',metrics=['acc'])
history = model.fit(train_features, train_labels,epochs=10,batch_size=64,validation_data=(test_features, test_labels))


import matplotlib.pyplot as plt
import cv2
img = cv2.imread('1.PNG')
cv2.imshow('image',img)
plt.imshow(img)
plt.show()

img2 = cv2.imread('2.PNG')
cv2.imshow('image',img2)
plt.imshow(img2)
plt.show()
from keras.models import load_model
import cv2
import numpy as np
model = load_model('VGG16_model.h5')

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

img = cv2.imread('1.PNG')
img = cv2.resize(img,(256,256))
img = np.reshape(img,[1,256,256,3])
xinput=conv_base.predict(img)
cc=model.predict_classes(xinput)
print(cc)


img2 = cv2.imread('2.PNG')
img2 = cv2.resize(img2,(256,256))
img2 = np.reshape(img2,[1,256,256,3])
xinput2=conv_base.predict(img2)
cc2=model.predict_classes(xinput2)
print(cc2)






