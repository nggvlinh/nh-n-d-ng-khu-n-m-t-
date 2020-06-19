from os import listdir
import cv2
from mtcnn.mtcnn import MTCNN
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras, one_hot
import tensorflow as tf
from sklearn.svm import SVC
from PIL import Image
import os
import matplotlib.pyplot as plt
import pickle
from keras.models import load_model

raw_folder ="data/simple/"
path = "data/handling/"


facenet_model = load_model('facenet_keras.h5')
detector = MTCNN()
dest_size = (160, 160)
print("Bắt đầu xử lý crop mặt...")


'''for folder in listdir(raw_folder):

  if not os.path.exists(path):
    os.mkdir(path)
  else :
    x_folder = path + folder
    print(x_folder)
    if not os.path.exists(x_folder):
        os.mkdir(x_folder)

    for file in listdir(raw_folder  + folder):
        print(file)
        image = Image.open(raw_folder + folder + "/" + file)
        image = image.convert('RGB')

        detector = MTCNN()
        pix = np.asarray(image)
        result = detector.detect_faces(pix)
        print(result)

        x1, y1, width, height = result[0]['box']
        x1, y1 = abs(x1), abs(y1)
        x2, y2 = x1 + width, y1 + height
        face = pix[y1:y2, x1:x2]
        image = Image.fromarray(face)
        image = image.resize(dest_size)

        plt.figure()
        plt.imshow(image[0])
        plt.colorbar()
        plt.grid(False)
        plt.show()
        print(folder)
        
        image.save(x_folder + "/" + file)
'''
# load image
'''
def get_embedding(model, face_pixels):
	# scale pixel values
	face_pixels = face_pixels.astype('float32')
	# standardize pixel values across channels (global)
	mean, std = face_pixels.mean(), face_pixels.std()
	face_pixels = (face_pixels - mean) / std
	# transform face into one sample
	samples = np.expand_dims(face_pixels, axis=0)
	# make prediction to get embedding
	yhat = model.predict(samples)
	return yhat[0]

def load_faces(train_folder ):
    if os.path.exists("faces_data.npz"):
        data = np.load('faces_data.npz')
        X_train,y_train = data["arr_0"],data["arr_1"]
        return X_train, y_train
    else:
        X_train = []
        y_train = []

        # enumerate folders, on per class
        for folder in listdir(train_folder):
            # Lặp qua các file trong từng thư mục chứa các em
            for file in listdir(train_folder + folder):
                # Read file
                image = Image.open(train_folder + folder + "/" + file)
                # convert to RGB, if needed
                image = image.convert('RGB')
                # convert to array
                pixels = np.asarray(image)

                # Thêm vào X
                X_train.append(pixels)
                y_train.append(folder)

        X_train = np.array(X_train)
        y_train = np.array(y_train)

        # Check dữ liệu
        print(X_train)
        plt.figure()
        plt.imshow(X_train[0])
        plt.colorbar()
        plt.grid(False)
        plt.show()

        print(X_train.shape)
        print(y_train.shape)
        print(y_train[0:2])

        # Convert du lieu y_train
        output_enc = LabelEncoder()
        output_enc.fit(y_train)
        y_train = output_enc.transform(y_train)
        pkl_filename = "output_enc.pkl"
        with open(pkl_filename, 'wb') as file:
            pickle.dump(output_enc, file)

        print(y_train[0:2])

        print(X_train.shape)
        print(y_train.shape)
        print(type(X_train))
        print(type(y_train))

        # Convert du lieu X_train sang embeding
        X_train_emb = []
        for x in X_train:
            X_train_emb.append( get_embedding(facenet_model, x))

        X_train_emb = np.array(X_train_emb)

        print("Load faces done!")
        # Save
        np.savez_compressed('faces_data.npz', X_train_emb, y_train);
        return X_train_emb, y_train


X , y = load_faces(path)

print(X.shape)
print(y.shape)


model = SVC(kernel='linear',probability=True)
model.fit(X, y)

pkl_filename = "faces_svm.pkl"
with open(pkl_filename, 'wb') as file:
    pickle.dump(model, file)

print("Saved model")

'''
def get_embedding(model, face_pixels):
	# scale pixel values
	face_pixels = face_pixels.astype('float32')
	# standardize pixel values across channels (global)
	mean, std = face_pixels.mean(), face_pixels.std()
	face_pixels = (face_pixels - mean) / std
	# transform face into one sample
	samples = np.expand_dims(face_pixels, axis=0)
	# make prediction to get embedding
	yhat = model.predict(samples)
	return yhat[0]


x_train = []
y_train = []
for f_1 in listdir(path):
    #print(f_1)
    for read_file in listdir(path + f_1):
           image = Image.open(path+ f_1 +'/'+read_file)
           image = image.convert('RGB')
           #x =path+ f_1 +'/'+read_file

           #img = cv2.imread(x)
           col_img = np.asarray(image)
           #col = np.array(f_1)

           x_train.append(col_img)
           y_train.append(f_1)



x_train=np.vstack([x_train])
y_train=np.vstack([y_train]).T


## encoder labels

labels = LabelEncoder()
labels.fit(y_train)
y_train = labels.transform(y_train)

pkl = "output_labels.pkl"
with open(pkl, 'wb') as file:
    pickle.dump(pkl, file)

## process data train
print("Trước khi training data để train svm")
print(x_train.shape)
print(y_train.shape)

x_train_embedding = []
for x in x_train :
    em = get_embedding(facenet_model,x)
    np.append(x_train_embedding,em)
    x_train_embedding = np.array(x_train_embedding)

print("Sau khi training data để train svm")
print(x_train_embedding.shape)
print(y_train.shape)

'''
print(x_train)
plt.figure()
plt.imshow(x_train[0])
plt.colorbar()
plt.grid(False)
plt.show()

print(x_train.shape)
print(y_train_label.shape)
print(type(x_train))
print(type(y_train_label))
'''

'''


model = keras.Sequential([keras.layers.Flatten(input_shape=(160,160)),\
        keras.layers.Dense(128,activation='relu'),\
        keras.layers.Dense(40)])

model.summary()

#optimizer='adam'
model.compile(optimizer='adam',\
        loss=tf.keras.losses.\
        SparseCategoricalCrossentropy(from_logits=True),\
        metrics=['accuracy'])

#huan luyên mô hình voi epochs=
model.fit(x_train,y_train_label,epochs=100)'''