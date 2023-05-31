import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from cutout import ct

# 数据准备
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()
# 转换one-hot独热映射
y_train = tf.keras.utils.to_categorical(y_train, num_classes=100)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=100)


# 数据增强
for i in range(50000):
    x_train[i] = ct(x_train[i],size=16)



# 预处理

x_train = x_train / 255.0
x_test = x_test / 255.0

# 构建模型
def vgg16D():
    model = tf.keras.Sequential()

    # 第一块
    # conv1
    model.add(
        tf.keras.layers.Conv2D(filters=64, kernel_size=[3, 3], input_shape=(32, 32, 3), strides=1, activation='relu',
                               padding='SAME', kernel_regularizer=tf.keras.regularizers.l2(0.0005)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.4))

    model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=[3, 3], strides=1, activation='relu',
                                     padding='SAME', kernel_regularizer=tf.keras.regularizers.l2(0.0005)))
    model.add(tf.keras.layers.BatchNormalization())
    # 最大池化层1
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

    # 第二块
    # conv3
    model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=[3, 3], strides=1, activation='relu',
                                     padding='SAME', kernel_regularizer=tf.keras.regularizers.l2(0.0005)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.4))
    # conv4
    model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=[3, 3], strides=1, activation='relu',
                                     padding='SAME', kernel_regularizer=tf.keras.regularizers.l2(0.0005)))
    model.add(tf.keras.layers.BatchNormalization())
    # 最大池化层2
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

    # conv5
    model.add(tf.keras.layers.Conv2D(filters=256, kernel_size=[3, 3], strides=1, activation='relu',
                                     padding='SAME', kernel_regularizer=tf.keras.regularizers.l2(0.0005)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.4))
    # conv6
    model.add(tf.keras.layers.Conv2D(filters=256, kernel_size=[3, 3], strides=1, activation='relu',
                                     padding='SAME', kernel_regularizer=tf.keras.regularizers.l2(0.0005)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.4))
    # conv7
    model.add(tf.keras.layers.Conv2D(filters=256, kernel_size=[3, 3], strides=1, activation='relu',
                                     padding='SAME', kernel_regularizer=tf.keras.regularizers.l2(0.0005)))
    model.add(tf.keras.layers.BatchNormalization())
    # 最大池化层3
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

    # conv8
    model.add(tf.keras.layers.Conv2D(filters=512, kernel_size=[3, 3], strides=1, activation='relu',
                                     padding='SAME', kernel_regularizer=tf.keras.regularizers.l2(0.0005)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.4))
    # conv9
    model.add(tf.keras.layers.Conv2D(filters=512, kernel_size=[3, 3], strides=1, activation='relu',
                                     padding='SAME', kernel_regularizer=tf.keras.regularizers.l2(0.0005)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.4))
    # conv10
    model.add(tf.keras.layers.Conv2D(filters=512, kernel_size=[3, 3], strides=1, activation='relu',
                                     padding='SAME', kernel_regularizer=tf.keras.regularizers.l2(0.0005)))
    model.add(tf.keras.layers.BatchNormalization())
    # 最大池化层4
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

    # conv11
    model.add(tf.keras.layers.Conv2D(filters=512, kernel_size=[3, 3], strides=1, activation='relu',
                                     padding='SAME', kernel_regularizer=tf.keras.regularizers.l2(0.0005)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.4))
    # conv12
    model.add(tf.keras.layers.Conv2D(filters=512, kernel_size=[3, 3], strides=1, activation='relu',
                                     padding='SAME', kernel_regularizer=tf.keras.regularizers.l2(0.0005)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.4))
    # conv13
    model.add(tf.keras.layers.Conv2D(filters=512, kernel_size=[3, 3], strides=1, activation='relu',
                                     padding='SAME', kernel_regularizer=tf.keras.regularizers.l2(0.0005)))
    model.add(tf.keras.layers.BatchNormalization())
    # 最大池化层5
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

    # f-c 三层
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dropout(rate=0.5))

    model.add(tf.keras.layers.Dense(units=512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0005)))
    model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.Dense(units=100))
    model.add(tf.keras.layers.Activation('softmax'))
    # 查看摘要
    model.summary()
    return model


# 超参设置
training_epochs = 100
batch_size = 512
learning_rate = 0.1
momentum = 0.9  # SGD加速动量
lr_decay = 1e-6  # 学习衰减
lr_drop = 20  # 衰减倍数
model = vgg16D()


# ——
# ——


# 每20个epoch 学习率缩小为原来的一半
def lr_scheduler(epoch):
    return learning_rate * (0.5 ** (epoch // lr_drop))


reduce_lr = tf.keras.callbacks.LearningRateScheduler(lr_scheduler)

# 使用sgd优化器
optimizer = tf.keras.optimizers.legacy.SGD(learning_rate=learning_rate,
                                           decay=1e-6, momentum=momentum, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

history = model.fit(x_train, y_train,batch_size=batch_size, epochs=training_epochs, callbacks=[reduce_lr],
                    steps_per_epoch=x_train.shape[0] // batch_size,
                    validation_data=(x_test, y_test))
# 绘制训练 && 验证的准确率
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# 绘制训练 && 验证的损失值
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

model.save('cutout.h5')
