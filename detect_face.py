from os import listdir
from mtcnn.mtcnn import MTCNN
import os
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from PIL import Image
import pickle
from keras.models import load_model

raw_folder ="data/simple/"
path = "data/handling/"

facenet_model = load_model('facenet_keras.h5')
detector = MTCNN()
dest_size = (160, 160)
print("Bắt đầu xử lý crop mặt...")


for folder in listdir(raw_folder):

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

        
        image.save(x_folder + "/" + file)

# load image

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
    pickle.dump(labels, file)

## process data train
print("Trước khi training data để train svm")
print(x_train.shape)
print(y_train.shape)

x_train_embedding = []
for x in x_train :
    emb = get_embedding(facenet_model, x)
    x_train_embedding.append(emb)

x_train_embedding = np.array(x_train_embedding)

print("Sau khi training data để train svm")
print(x_train_embedding.shape)
print(y_train.shape)

model = SVC(kernel='linear',probability=True)
model.fit(x_train_embedding, y_train)

pkl_filename = "train_svm.pkl"
with open(pkl_filename, 'wb') as file:
    pickle.dump(model, file)
