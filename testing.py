from keras.models import load_model
from keras.datasets import cifar100
from keras.utils import to_categorical
import numpy as np
import random

# 加载 cifar100 数据集
(x_train, y_train), (x_test, y_test) = cifar100.load_data(label_mode='fine')

# 数据预处理：将输入图像像素值归一化到0-1之间，将标签进行 one-hot 编码
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
y_train = to_categorical(y_train, num_classes=100)
y_test = to_categorical(y_test, num_classes=100)

# 加载训练好的 VGG
model = load_model('cutmix.h5')

# 从测试集中随机抽取一张图像并进行预测
index = random.randint(0, len(x_test)-1)
x_sample = x_test[index]
y_true = y_test[index]
y_pred = model.predict(np.expand_dims(x_sample, axis=0))

# 输出预测结果和真实类别标签
print('Predicted label:', np.argmax(y_pred))
print('True label:', np.argmax(y_true))