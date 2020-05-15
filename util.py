import tensorflow as tf
import matplotlib.pyplot as plt

global args


def image_show(image, label):
    plt.figure()
    plt.imshow(image)
    plt.title(label)
    plt.axis('off')
    plt.show()
    plt.close()