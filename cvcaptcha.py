import numpy as np # linear algebra
import pandas as pd
#%matplotlib inline 
import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from keras import layers
from keras.models import Model
from keras.models import load_model
from keras import callbacks
import cv2
import string

#Init main values
symbols = string.ascii_lowercase + "0123456789" # All symbols captcha can contain
num_symbols = len(symbols)
img_shape = (50, 200, 1)

def create_model():
    img = layers.Input(shape=img_shape) # Get image as an input and process it through some Convs
    conv1 = layers.Conv2D(16, (3, 3), padding='same', activation='relu')(img)
    mp1 = layers.MaxPooling2D(padding='same')(conv1)  # 100x25
    conv2 = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(mp1)
    mp2 = layers.MaxPooling2D(padding='same')(conv2)  # 50x13
    conv3 = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(mp2)
    bn = layers.BatchNormalization()(conv3)
    mp3 = layers.MaxPooling2D(padding='same')(bn)  # 25x7
    
    # Get flattened vector and make 5 branches from it. Each branch will predict one letter
    flat = layers.Flatten()(mp3)
    outs = []
    for _ in range(5):
        dens1 = layers.Dense(64, activation='relu')(flat)
        drop = layers.Dropout(0.5)(dens1)
        res = layers.Dense(num_symbols, activation='softmax')(drop)

        outs.append(res)
    
    # Compile model and return it
    model = Model(img, outs)
    model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=["accuracy"])
    return model


def modelexist():
    if os.path.isfile("./captcha.h5"):
        return True


def preprocess_data():
    n_samples = len(os.listdir('./input/captcha-version-2-images'))
    X = np.zeros((n_samples, 50, 200, 1)) #1070*50*200
    y = np.zeros((5, n_samples, num_symbols)) #5*1070*36

    for i, pic in enumerate(os.listdir('./input/captcha-version-2-images')):
        # Read image as grayscale
        img = cv2.imread(os.path.join('./input/captcha-version-2-images', pic), cv2.IMREAD_GRAYSCALE)
        pic_target = pic[:-4]
        if len(pic_target) < 6:
            # Scale and reshape image
            img = img / 255.0
            img = np.reshape(img, (50, 200, 1))
            # Define targets and code them using OneHotEncoding
            targs = np.zeros((5, num_symbols))
            for j, l in enumerate(pic_target):
                ind = symbols.find(l)
                targs[j, ind] = 1
            X[i] = img
            y[:, i] = targs
    
    # Return final data
    return X, y


    
#Create a Drawing Function
def show_train_history(train_history,train,validation,label):
    #Write the code
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title("Train History")
    plt.ylabel(label)
    plt.xlabel("Epoch")
    plt.legend(['train','validation'],loc='upper left')
    plt.show()
    
def build_model(bsize,eps):
    X, y = preprocess_data()
    X_train, y_train = X[:970], y[:, :970]
    X_test, y_test = X[970:], y[:, 970:]
    model=create_model();
    #hist = model.fit(X_train, [y_train[0], y_train[1], y_train[2], y_train[3], y_train[4]], batch_size=32, epochs=30,verbose=1, validation_split=0.2)
    hist = model.fit(X_train, [y_train[0], y_train[1], y_train[2], y_train[3], y_train[4]], batch_size=bsize, epochs=eps,verbose=1, validation_split=0.2)
    score= model.evaluate(X_test,[y_test[0], y_test[1], y_test[2], y_test[3], y_test[4]],verbose=1)
    print('Test Loss and accuracy:', score)
    show_train_history(hist,'dense_9_accuracy','val_dense_9_accuracy','accuracy')
    show_train_history(hist,'dense_9_loss','val_dense_9_loss','loss')
    return model
    
def savemodel(newmodel):
    newmodel.save('captcha.h5')

def predict(filepath,usenewmodel):
    img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    if img is not None:
        img = img / 255.0
    else:
        print("Not detected");
    if(usenewmodel==0):
        mymodel= load_model('./defaultcaptcha.h5')
    else:
        mymodel= load_model('./captcha.h5')
    res = np.array(mymodel.predict(img[np.newaxis, :, :, np.newaxis]))
    ans = np.reshape(res, (5, 36))
    l_ind = []
    probs = []
    for a in ans:
        l_ind.append(np.argmax(a))
        #probs.append(np.max(a))

    capt = ''
    for l in l_ind:
        capt += symbols[l]
    return capt#, sum(probs) / 5

# import matplotlib.pyplot as plt
# print("Predicted Captcha =",predict('./input/capthaimages/a.png'))
# img=cv2.imread('./input/capthaimages/a.png',cv2.IMREAD_GRAYSCALE)
# imgplot=plt.imshow(img)
# plt.show()