import glob
import os
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from PIL import Image
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")


#At this part, we're going to train our neural networks with dataset that prepared by us as well.
#I will explain my algorithm to you as much as I can.



imgs = glob.glob("./img/*.png")

ArithmeticError()
width = 125
height = 50

X = []
Y = []



#At this part, we're getting images from their files. Splitting them due to differences and we can do this with splitting them via "_"
#Also we're changing the size of the images here just for decreasing the time of training.
#Here we are converting those images into np arrays because training can just be with arrays.
#We are dividing this array with 255 for ranging numbers between 0 and 1.
for img in imgs:
    
    filename = os.path.basename(img)
    label = filename.split("_")[0]
    im = np.array(Image.open(img).convert("L").resize((width, height)))
    im = im / 255
    X.append(im)
    Y.append(label)
    
X = np.array(X)
X = X.reshape(X.shape[0], width, height, 1)

# sns.countplot(Y)

def onehot_labels(values):
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(values)
    onehot_encoder = OneHotEncoder(sparse = False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded),1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    return onehot_encoded

Y = onehot_labels(Y)
train_X, test_X, train_y, test_y = train_test_split(X, Y , test_size = 0.25, random_state = 2)    

# cnn model
# We are creating our cnn model here and adding features to our neural network.
# If you're wondering how they works one by one, you can search it on kaggle, stackoverflow etc.
model = Sequential()   
model.add(Conv2D(32, kernel_size = (3,3), activation = "relu", input_shape = (width, height, 1)))
model.add(Conv2D(64, kernel_size = (3,3), activation = "relu"))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation = "relu"))
model.add(Dropout(0.4))
model.add(Dense(2, activation = "softmax"))


model.compile(loss = "categorical_crossentropy", optimizer = "Adam", metrics = ["accuracy"])

#epochs equals the number of stage of training.
#Increasing epochs will be more beneficial for training but there is a thing called as overfitting
#If epochs greater than your model needs, your model will be overfitted and won't be useful.
#So be aware and also research this. In addition, increasing epochs will increase training time as well.
model.fit(train_X, train_y, epochs = 35, batch_size = 64)

score_train = model.evaluate(train_X, train_y)
print("Training Accuracy: %",score_train[1]*100)
    
score_test = model.evaluate(test_X, test_y)
print("Test Accuracy: %",score_test[1]*100)
    
#With all of those codes, we are getting a trained model to use:
open("model_new.json","w").write(model.to_json())
model.save_weights("agackesme.h5")   
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    