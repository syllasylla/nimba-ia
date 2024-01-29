import os
import random
import string
from argparse import ArgumentParser
# For measuring inference time
from timeit import default_timer as timer

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.activations import softmax

from custom import cat_acc, cce, plate_acc, top_3_k

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


def check_low_conf(probs, thresh=.3):
    return [i for i, prob in enumerate(probs) if prob < thresh]


@tf.function
def predict_from_array(img, model):
    pred = model(img, training=False)
    return pred


def probs_to_plate(prediction):
    prediction = prediction.reshape((7, 37))
    probs = np.max(prediction, axis=-1)
    prediction = np.argmax(prediction, axis=-1)
    plate = list(map(lambda x: alphabet[x], prediction))
    return plate, probs


def visualize_predictions(model, imgs_path):
    # generate samples and plot
    val_imgs = [os.path.join(imgs_path, f) for f in os.listdir(imgs_path)
                if os.path.isfile(os.path.join(imgs_path, f))]

    for im in val_imgs:

        im = cv2.imread(im, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(im, dsize=(140, 70), interpolation=cv2.INTER_LINEAR)
        img = img[np.newaxis, ..., np.newaxis] / 255.
        img = tf.constant(img, dtype=tf.float32)

        prediction = predict_from_array(img, model).numpy()
        plate, probs = probs_to_plate(prediction)
        plate_str = ''.join(plate)
        print(plate_str, flush=True)

        im_to_show = cv2.resize(im, dsize=(140 * 3, 70 * 3), interpolation=cv2.INTER_LINEAR)
        im_to_show = cv2.cvtColor(im_to_show, cv2.COLOR_GRAY2RGB)

        cv2.imshow(f'plates', im_to_show)
        if cv2.waitKey(0) & 0xFF == ord('q'):
            return


custom_objects = {
    'cce': cce,
    'cat_acc': cat_acc,
    'plate_acc': plate_acc,
    'top_3_k': top_3_k,
    'softmax': softmax
}

model_path = './models/m3_91_vpc_1.3M_CPU.h5'
img_dir = './plaque'

alphabet = string.digits + string.ascii_uppercase + '_'
model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)

visualize_predictions(model, img_dir)

cv2.destroyAllWindows()
