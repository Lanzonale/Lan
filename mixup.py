import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()

y_train = tf.keras.utils.to_categorical(y_train, num_classes=100)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=100)


def mixup(image, label, alpha):
    index = np.random.choice(len(x_train))
    x2 = x_train[index]
    y2 = y_train[index]
    mix_label = label * alpha + y2 * (1 - alpha)
    mix_image = image * alpha + x2 * (1 - alpha)
    return mix_image, mix_label



# 对图像和标签进行 MixUp
alpha = 0.3
mix_image, mix_label = mixup(x_train[9], y_train[9], alpha)

# 绘制原图像和 MixUp 后的图像
plt.subplot(1, 2, 1)
plt.imshow(x_train[9]/255)
plt.title('Original image')
plt.subplot(1, 2, 2)
plt.imshow(mix_image/255)
plt.title('MixUp image')
plt.show()

