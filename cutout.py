import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from numpy import dtype

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()

y_train = tf.keras.utils.to_categorical(y_train, num_classes=100)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=100)
def cutout(image, size):
    h, w, c = image.shape
    pad = size // 2
    x = np.random.randint(pad, h - pad)
    y = np.random.randint(pad, w - pad)
    mask = np.ones((h, w, c), dtype('uint8'))
    mask[x-pad:x+pad, y-pad:y+pad, :] = 0.
    image *= mask
    return image

# 随机选取一张图像

# 对图像进行 CutOut

print(y_test[2775])
# 绘制原图像和 CutOut 后的图像

plt.imshow(x_test[2775]/255)
plt.show()




