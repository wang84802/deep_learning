import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random
import tensorflow as tf
import keras_preprocessing
import config
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

x = []
y = []
IMG_SIZE = 150

folders = os.listdir(config.DIR)

# 資料分類
for i, file in enumerate(folders):
    filename = os.path.join(config.DIR, file)
    for img in os.listdir(filename):
        path = os.path.join(filename, img)
        img = cv2.imread(path,cv2.IMREAD_COLOR)
        img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))

        x.append(np.array(img))
        y.append(i)

np.unique(y, return_counts=True)

x = np.array(x)
y = np.array(y)

print("X shape is {}".format(x.shape))
print("y shape is {}".format(y.shape))

# preprocessing data
y = np_utils.to_categorical(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=45)

# image data augmentation 資料增強=>加強CNN辨識率
training_datagen = keras_preprocessing.image.ImageDataGenerator(
      rescale = 1./255,
      rotation_range=40,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')

training_datagen.fit(x_train)

validation_datagen = keras_preprocessing.image.ImageDataGenerator(
      rescale = 1./255)

validation_datagen.fit(x_test)

model = tf.keras.models.Sequential([
    # 第一層convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(150, 150, 3)),
    # 找2*2裡最大
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # 變成一維陣列
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.5),
    # 512 neuron hidden layer
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(5, activation='softmax')
])

model.summary()

model.compile(loss = 'categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

batch_size=32

# fit generator只給部分的資料 重複更新直到epoch數量
history = model.fit_generator(training_datagen.flow(x_train,y_train, batch_size=batch_size),
                              epochs = 15, validation_data = validation_datagen.flow(x_test, y_test, batch_size=batch_size),
                              verbose = 1, steps_per_epoch=x_train.shape[0] // batch_size)

predictions = model.predict(x_test)
prediction_digits = np.argmax(predictions, axis=1)

real_labels = np.argmax(y_test, axis=1)

from sklearn.metrics import confusion_matrix

print()
print('test data confusion matrix:')
print(confusion_matrix(real_labels, prediction_digits))

print()
print('test data Performance metrics:')
print(classification_report(real_labels, prediction_digits))


# 預測(prediction)
r_num = random.randint(0,len(x_test)-1)
X = x_test[r_num:r_num + 10,:] #random取10個元素
predictions = np.argmax(model.predict(X), axis=-1) #返回最大索引值

# 預測值
print('predictions: '+ str(predictions))

# 正確值
plt.figure(figsize=(10,10))
for i in range(10):
    plt.subplot(5,5,i+1)
    plt.imshow(cv2.cvtColor(x_test[r_num], cv2.COLOR_BGR2RGB))
    plt.title(np.argmax(y_test[r_num], axis=-1))
    r_num += 1
    plt.tight_layout()
plt.show()

acc = history.history['accuracy'] # 訓練集
val_acc = history.history['val_accuracy'] # 測試集
loss = history.history['loss'] # 訓練集
val_loss = history.history['val_loss'] # 測試集

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')

plt.figure()

plt.plot(epochs, loss, 'r', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()