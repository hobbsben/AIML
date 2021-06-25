# Import TensorFlow 
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub

# TODO: Make all other necessary imports.
import numpy as np
import matplotlib.pyplot as plt
import json as json
from tensorflow import keras
from tensorflow.keras.models import load_model
from PIL import Image
import argparse
import sys
import os
def process_image(images):
    im = tf.convert_to_tensor(images)
    im = tf.image.resize_with_crop_or_pad(im, target_height = 224, target_width = 224)
    im = tf.cast(im, tf.float32)
    im /= 255
#     im = np.asarray(im)
    
    return im
def predict(image_path,model,top_k):
#     im = Image.open(image_path)
#     test_image = np.asarray(im)
    processed_test_image = process_image(image_path)
    ps = model.predict( np.expand_dims(processed_test_image,axis = 0))
#     probs = np.argmax(ps,axis = 0)
#     probs = ps.argmax(ps)
    top_k_values, top_k_indices = tf.nn.top_k(ps, k=top_k)
    top_k_indices = 1 + top_k_indices[0]
    return top_k_values.numpy()[0], top_k_indices.numpy().astype(str)


# # for image_batch, label_batch in testing_batches.take(1):
# #     ps = model.predict(image_batch)
# #     images = image_batch.numpy().squeeze()
# #     labels = label_batch.numpy()

# for n in range(num_samples):
#     plt.subplot(6,5,n+1)
#     color = 'green' if np.argmax(ps[n]) == label_batch.numpy()[n] else 'red'
#     plt.imshow(image_batch.numpy().squeeze()[n], cmap = plt.cm.binary)
#     plt.title(class_names[str(np.argmax(ps[n]))],color = str(color))


parser = argparse.ArgumentParser(description='image classifier')
parser.add_argument('image_path', type=str, help='Input dir for images')
parser.add_argument('saved_model',type=str, help='Input dir for model')
parser.add_argument('--top_k', dest="top_k", type=int)
parser.add_argument('--category_names', dest="class_names")
args = parser.parse_args()

im = Image.open(args.image_path)
test_image = np.asarray(im)

# TODO: Load the Keras model
reloaded_keras_model  = load_model("model.h5",custom_objects={'KerasLayer':hub.KerasLayer}, compile = False)
reloaded_keras_model.summary()

# processed_image = process_image(test_image)
probs, classes = predict(test_image, reloaded_keras_model, args.top_k)
print(probs)

print ( np.argmax(probs))
with open(args.class_names, 'r') as f:
     class_names = json.load(f)
print( classes)
print(class_names[classes[np.argmax(probs)]])

class_name_array = [class_names[classes[0]],class_names[classes[1]],class_names[classes[2]],class_names[classes[3]],class_names[classes[4]]]
print(class_name_array)

