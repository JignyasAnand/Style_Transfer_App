import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tensorflow_hub as hub
import os
from matplotlib import gridspec

class CONFIG:
  output_image_size = 384
  content_img_size = (output_image_size, output_image_size)
  style_img_size = (256, 256)

class HUB_NST:
  def __init__(self):
    hub_handle = 'https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2'
    self.hub_module = hub.load(hub_handle)



  def get_stylized_image(self, style_image, content_image):
    style_image = load_cropped_image(style_image, CONFIG.style_img_size, crop=True, fixed=True)
    content_image = load_cropped_image(content_image, CONFIG.content_img_size, crop=False, fixed=True)
    style_image = tf.nn.avg_pool(style_image, ksize=[3, 3], strides=[1, 1], padding='SAME')
    show_n([style_image, content_image])
    outputs = self.hub_module(tf.constant(content_image), tf.constant(style_image))
    return outputs[0]


def crop_center(img):
  shape = img.shape
  new_shape = min(shape[1], shape[2])
  offset_y = max(shape[1]-shape[2], 0)
  offset_x = max(shape[2]-shape[1], 0)
  img = tf.image.crop_to_bounding_box(
      img, offset_y, offset_x, new_shape, new_shape
  )
  return img
def expand_img(img_arr):
  img = tf.expand_dims(
      img_arr, 0
  )
  return img

def load_cropped_image(img_arr, img_size=(256, 256), crop=False, fixed=False):
  img = expand_img(img_arr)
  if crop:
    img = crop_center(img)
  if fixed:
    temp = tf.keras.layers.experimental.preprocessing.Rescaling(1. / 255)
    img = tf.image.resize(img, img_size, preserve_aspect_ratio=True)
    img = temp(img)

  return img

def show_n(images, titles=('',)):
  n = len(images)
  image_sizes = [image.shape[1] for image in images]
  w = (image_sizes[0] * 6) // 320
  plt.figure(figsize=(w * n, w))
  gs = gridspec.GridSpec(1, n, width_ratios=image_sizes)
  for i in range(n):
    plt.subplot(gs[i])
    plt.imshow(images[i][0], aspect='equal')
    plt.axis('off')
    plt.title(titles[i] if len(titles) > i else '')
  plt.show()




