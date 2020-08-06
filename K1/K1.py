import numpy as np 
import tensorflow as tf
import keras
from keras.layers import Conv2D, Dense, MaxPool2D, BatchNormalization, Activation, Flatten
from keras.models import Input, Model
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt 
import seaborn as sns
sns.set()
np.random.seed(2)

# load data set
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_cv, x_test = train_test_split(x_test, test_size = 0.5, shuffle=False)
y_cv, y_test = train_test_split(y_test, test_size = 0.5, shuffle=False)

# create the model
def MNIST_Model(input_shape):
    X_input = Input(input_shape)

    X = tf.expand_dims(X_input, -1)
    
    X = Conv2D(28, (7,7), padding='same', name='conv0')(X)
    X = BatchNormalization(axis=-1, name='bn0')(X)
    X = Activation('relu')(X)

    X = MaxPool2D(pool_size=(2,2), name='max_pool0')(X)

    X = Conv2D(28, (7,7), padding='same', name='conv1')(X)
    X = BatchNormalization(axis=-1, name='bn1')(X)
    X = Activation('relu')(X)

    X = MaxPool2D(pool_size=(2,2), name='max_pool1')(X)

    X = Conv2D(64, (7,7), padding='same', name='conv2')(X)
    X = BatchNormalization(axis=-1, name='bn2')(X)
    X = Activation('relu')(X)

    X = MaxPool2D(pool_size=(2,2), name='max_pool2')(X)

    X = Flatten()(X)
    X = Dense(10, activation='softmax', name='fc')(X)

    model = Model(inputs = X_input, outputs = X, name='MyModel')

    return model

# train the model
model = MNIST_Model(x_test.shape[1:])
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
y_train = keras.utils.to_categorical(y_train)
model.fit(x_train,y_train)

#ready for prediction and answer
percentage = model.predict(x_test)
m = x_test.shape[0]
print(m)
prediction = np.empty(m)
answer = np.empty(m)
for i in range(m):
    prediction[i] = np.where(percentage[i]==max(percentage[i]))[0][0]
    answer[i] = np.where(y_test[i]==1.0)[0][0]

for i in range(10):
    print(prediction[i], answer[i])

# ready for evaluate this model
def evaluate_model(TP, FP, TN, FN, checker, prediction, answer):
    if answer == checker and prediction == checker:
        TP += 1
    elif answer == checker and prediction != checker:
        TN += 1
    elif answer != checker and prediction == checker:
        FP += 1
    else:
        FN += 1
    return TP, FP, TN, FN

# calculate each score
for i in range(10):
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    for j in range(m):
        TP, FP, TN, FN = evaluate_model(TP, FP, TN, FN, i, prediction[j], answer[j])
    p_score = TP / (TP + FP)
    r_score = TP / (TP + FN)
    f_score = 2 * (r_score * p_score) / (r_score + p_score)
    print('{}: precision = {:.4f}, recall = {:.4f}, f-score = {:.4f}'.format(i, p_score, r_score, f_score))