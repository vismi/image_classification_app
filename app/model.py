import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.preprocessing import image
from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import pandas as pd
from keras.utils import np_utils
import numpy as np

def load_model():
    model = build_model()
    model = train_model(model)
    return model

def build_model():
    model = Sequential()
    model.add(Dense(8, input_dim=4, activation='relu'))
    model.add(Dense(3, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def train_model(model):
    iris = load_iris()
    X = iris.data
    y = iris.target

    encoder = LabelEncoder()
    encoder.fit(y)
    encoded_y = encoder.transform(y)
    y_train = np_utils.to_categorical(encoded_y)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X)

    model.fit(X_train, y_train, epochs=100, batch_size=10)

    return model

def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0
    return img

def classify_image(model, input_data):
    predictions = model.predict(pd.DataFrame([input_data]))
    classes = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
    predicted_class = classes[np.argmax(predictions)]
    confidence = np.max(predictions)
    print(predicted_class, confidence)
    return predicted_class, confidence