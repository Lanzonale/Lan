import numpy as np
import tensorflow as tf

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()

y_train = tf.keras.utils.to_categorical(y_train, num_classes=100)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=100)


def cm(image, label, PROBABILITY, DIM):
    # 样本数量
    n_classes = label.shape[0]

    # 随机判断是否应用 CutMix
    if np.random.random() < PROBABILITY:
        # 随机选取第二个样本
        cut_image = x_train[np.random.randint(0, x_train.shape[0]), :, :, :]
        cut_label = y_train[np.random.randint(0, n_classes), :]

        cut_label = np.expand_dims(cut_label, axis=0)

        # 随机选取两个比例因子
        lam = np.random.beta(1, 1)
        cut_ratio = int(DIM * np.sqrt(1.0 - lam))
        cx, cy = np.random.randint(0, DIM, size=2)
        bbx1, bby1, bbx2, bby2 = np.clip(
            [cx - cut_ratio // 2, cy - cut_ratio // 2, cx + cut_ratio // 2, cy + cut_ratio // 2], 0, DIM)

        # 对第一个样本应用 CutMix
        image1 = image.copy()
        image1[bbx1:bbx2, bby1:bby2, :] = cut_image[bbx1:bbx2, bby1:bby2, :]
        label1 = label.copy()
        label1 = label1*(1-(bbx2 - bbx1) * (bby2 - bby1)/1024)+cut_label*(bbx2 - bbx1) * (bby2 - bby1)/1024

        # 对第二个样本应用 CutMix
        image2 = cut_image.copy()
        image2[bbx1:bbx2, bby1:bby2, :] = image[bbx1:bbx2, bby1:bby2, :]
        label2 = cut_label.copy()
        label2 = lam * label + (1 - lam) * label2

        # 混合两个样本
        return image1, label1

    else:
        return image, label
