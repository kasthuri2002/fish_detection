
import os
folder_path = 'datasets/Fish_Dataset'

try:
    files = os.listdir(folder_path)
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp']
    image_files = [file for file in files if os.path.splitext(file)[1].lower() in image_extensions]
    print("Number of images in the folder:", len(image_files))
except FileNotFoundError:
    print("The specified directory does not exist.")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import seaborn
import wandb
from wandb.keras import WandbCallback
from PIL import Image
from tqdm import tqdm
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import seaborn as sns
from scipy.sparse import csr_matrix 
from sklearn.metrics import classification_report, confusion_matrix

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")
os.environ['KAGGLE_CONFIG_DIR'] = "../input/a-large-scale-fish-dataset"
# Replace the line plt.style.use("seaborn-dark") with the following code:
sns.set_style("dark")

sns.set_context("paper", font_scale=1.4)

BATCH_SIZE = 64
EPOCHS = 15
LEARNING_RATE = 0.001
RANDOM_STATE = 42
LOSS = "categorical_crossentropy"
OPTIMIZER = "adam"
METRICS = [
    "accuracy",
    "precision",
    "recall"
]

def get_dataset():
  main_directory = "datasets/Fish_Dataset"
  images = []
  labels = []
  for directory in tqdm(os.listdir(main_directory)):
      next_directory = f"{main_directory}/{directory}"
      if directory in ["README.txt", "license.txt", "Segmentation_example_script.m"]:
        continue
      for images_directory in os.listdir(next_directory):
          print(images_directory)
          if "GT" not in images_directory:
              final_directory = f"{next_directory}/{images_directory}"
              for image in os.listdir(final_directory):
                  images.append(np.array(Image.open(f"{final_directory}/{image}").resize((224, 224))))
                  labels.append(images_directory)
  images = np.array(images)
  labels = np.array(labels)
  return images, labels

images, labels = get_dataset()
print(images.shape)
print(labels.shape)

def plot_training_images(images, labels):
  plot_images = []
  plot_labels = []
  for i, j in zip(images, labels):
    if j in plot_labels:
      continue
    else:
      plot_images.append(i)
      plot_labels.append(j)
  fig, axes = plt.subplots(nrows = 3, ncols = 3, sharex=False, figsize=(12, 12))
  for i in range(3):
    for j in range(3):
      axes[i][j].imshow(plot_images[i * 3 + j])
      axes[i][j].set_xlabel(plot_labels[i * 3 + j])
      axes[i][j].set_xticks([])
      axes[i][j].set_yticks([])
  plt.tight_layout()
  plt.show()
# Count the number of unique labels or directories

num_classes = 16

def get_tf_dataset(images, labels):
    num_classes = len(np.unique(labels))
    if isinstance(labels, csr_matrix):
        labels = labels.toarray()
    labels = tf.one_hot(labels, depth=num_classes)  # Convert labels to one-hot encoding
    dataset = tf.data.Dataset.from_tensor_slices((images, labels)).batch(BATCH_SIZE).prefetch(1)
    return dataset  # Add this line to return the dataset

    

def split_dataset(images, labels, test_size = 0.2, valid_size = 0.2):
  train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size = test_size, random_state = RANDOM_STATE)
  train_images, valid_images, train_labels, valid_labels = train_test_split(train_images, train_labels, test_size = valid_size, random_state = RANDOM_STATE)
  return (train_images, train_labels), (valid_images, valid_labels), (test_images, test_labels)


def plot_cm(test_labels, prediction_labels, encoder):
  plt.figure(figsize=(15, 15))
  cm = confusion_matrix(test_labels, prediction_labels)
  df_cm = pd.DataFrame(cm, index = [i for i in encoder.categories_[0]],
                    columns = [i for i in encoder.categories_[0]])
  sns.set(font_scale=1.4)
  sns.heatmap(df_cm, annot=True, annot_kws={"size": 16}, fmt='g')
  plt.show()


def plot_history(history):
  fig, axes = plt.subplots(nrows = 2, ncols = 2, figsize=(20, 10))
  # Training
  sns.lineplot(x=range(1, len(history.history["loss"]) + 1),y=history.history["loss"], ax = axes[0][0])
  sns.lineplot(x=range(1, len(history.history["loss"]) + 1),y=history.history["accuracy"], ax = axes[0][1])
  sns.lineplot(x=range(1, len(history.history["loss"]) + 1),y=history.history["precision"], ax = axes[1][0])
  sns.lineplot(x=range(1, len(history.history["loss"]) + 1),y=history.history["recall"], ax = axes[1][1])
  # Validation
  sns.lineplot(x=range(1, len(history.history["loss"]) + 1),y=history.history["val_loss"], ax = axes[0][0])
  sns.lineplot(x=range(1, len(history.history["loss"]) + 1),y=history.history["val_accuracy"], ax = axes[0][1])
  sns.lineplot(x=range(1, len(history.history["loss"]) + 1),y=history.history["val_precision"], ax = axes[1][0])
  sns.lineplot(x=range(1, len(history.history["loss"]) + 1),y=history.history["val_recall"], ax = axes[1][1])

  axes[0][0].set_title("Loss Comparison", fontdict = {'fontsize': 20})
  axes[0][0].set_xlabel("Epoch")
  axes[0][0].set_ylabel("Loss")

  axes[0][1].set_title("Accuracy Comparison", fontdict = {'fontsize': 20})
  axes[0][1].set_xlabel("Epoch")
  axes[0][1].set_ylabel("Accuracy")

  axes[1][0].set_title("Precision Comparison", fontdict = {'fontsize': 20})
  axes[1][0].set_xlabel("Epoch")
  axes[1][0].set_ylabel("Precision")

  axes[1][1].set_title("Recall Comparison", fontdict = {'fontsize': 20})
  axes[1][1].set_xlabel("Epoch")
  axes[1][1].set_ylabel("Recall")
  plt.tight_layout()
  plt.show()


def get_resnet(categories):
  conv_block = tf.keras.applications.resnet.ResNet50(include_top = False, weights = "imagenet")
  output = tf.keras.layers.GlobalAveragePooling2D()(conv_block.output)
  output = tf.keras.layers.Dense(categories, activation = "softmax")(output)
  model = tf.keras.Model(inputs = [conv_block.input], outputs = [output])
  return model, "ResNet50"

def plot_images(images, true_labels, predicted_labels):
    plt.figure(figsize=(25, 15))
    for i in range(min(25, len(images))):  # Plot up to 25 images
        plt.subplot(5, 5, i + 1)
        plt.imshow(images[i], cmap='gray')  # Assuming images are grayscale
        plt.title(f'True: {true_labels[i]}\nPredicted: {predicted_labels[i]}')
        plt.axis('off')
    plt.show()

images, labels = get_dataset()
(train_images, train_labels), (valid_images, valid_labels), (test_images, test_labels) = split_dataset(images, labels, test_size=0.2, valid_size=0.2)
encoder = OneHotEncoder(dtype='float32')
train_labels = encoder.fit_transform(train_labels.reshape(-1, 1))
valid_labels = encoder.transform(valid_labels.reshape(-1, 1))
test_labels = encoder.transform(test_labels.reshape(-1, 1))
train_dataset = get_tf_dataset(train_images, train_labels)
valid_dataset = get_tf_dataset(valid_images, valid_labels)
train_dataset = get_tf_dataset(test_images, test_labels)

import matplotlib.pyplot as plt

def plot_training_images(images, labels):
    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    for i in range(3):
        for j in range(3):
            index = i * 3 + j
            if index < len(images):
                axes[i][j].imshow(images[index])
                axes[i][j].set_xlabel(labels[index])
                axes[i][j].set_xticks([])
                axes[i][j].set_yticks([])
    plt.tight_layout()
    plt.show()

plot_training_images(train_images, encoder.inverse_transform(train_labels).reshape(-1,))

model, model_name = get_resnet(len(encoder.categories_[0]))
config_defaults = {
  "learning_rate": LEARNING_RATE,
  "epochs": EPOCHS,
  "batch_size": BATCH_SIZE,
  "model_name": model_name,
  "loss": LOSS,
  "random_state": RANDOM_STATE,
  "optimizer": OPTIMIZER,
  "metrics": METRICS
}
#wandb.init(project="Fish_Dataset_Classification", id="resnet50", config = config_defaults)
model.compile(loss = LOSS, optimizer = OPTIMIZER, metrics = ["accuracy", tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])

history = model.fit(train_dataset,validation_data = valid_dataset,batch_size = BATCH_SIZE,epochs = EPOCHS,callbacks = [tf.keras.callbacks.EarlyStopping(monitor = "val_accuracy", patience = 5, restore_best_weights = True)])

model.save("models/fishimgmodel50.h5")

result_inter = model.predict(test_images)
prediction_index = np.argmax(result_inter, axis = -1)
result = np.zeros(shape = test_labels.shape, dtype = test_labels.dtype)
for i in range(result.shape[0]):
  result[i][prediction_index[i]] = 1.0

test_labels = encoder.inverse_transform(test_labels)
prediction_labels = encoder.inverse_transform(result)

plot_images(test_images, test_labels, prediction_labels)

print(classification_report(test_labels, prediction_labels))

#plot_cm(test_labels, prediction_labels, encoder)
prediction_indices = np.argmax(prediction_labels, axis=1)

plot_history(history)

cnn=tf.keras.models.load_model('models/fishimgmodel50.h5')

import cv2
image_path="datasets/Fish_Dataset/Fish_Dataset/Black Sea Sprat/Black Sea Sprat/00001.png"
img=cv2.imread(image_path)
plt.imshow(img)
plt.title("Test Image")
plt.xticks([])
plt.yticks([])
plt.show()

image=tf.keras.preprocessing.image.load_img(image_path)
input_arr=tf.keras.preprocessing.image.img_to_array(image)
input_arr=np.array([input_arr])
prediction=cnn.predict(input_arr)

import cv2
image_path="datasets/Fish_Dataset/Fish_Dataset/Black Sea Sprat/Black Sea Sprat/00001.png"
image=tf.keras.preprocessing.image.load_img(image_path)
input_arr=tf.keras.preprocessing.image.img_to_array(image)
input_arr=np.array([input_arr])
prediction=cnn.predict(input_arr)
max_probability = np.max(prediction) * 100
print("Maximum prediction probability:", max_probability)

categories = ['Black Sea Spart', 'Gilt Head Bream','Horse Mackerel',
'Red Sea Bream','Sea Bass','Shrimp','Striped Red Mullet','Trout']


# Get the index of the class with the maximum probability
predicted_class_index = np.argmax(prediction)
print("Predicted class index:",categories[predicted_class_index])