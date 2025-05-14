# 捕捉人脸训练模型
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.python.data import AUTOTUNE

from tensorflow.keras import layers


def fitModel(img, label):
    # 先划分数据集，按照8:1
    img_t, img_v, label_t, label_v = train_test_split(img, label, test_size=0.2, random_state=42)

    # 划分批次、混淆、缓冲区，来自张量作为数据管道
    t_ds = tf.data.Dataset.from_tensor_slices((img_t, label_t)).shuffle(1000).batch(32)
    v_ds = tf.data.Dataset.from_tensor_slices((img_v, label_v)).shuffle(1000)

    # 处理图像，归一化标准化，然后启用并行处理
    t_ds = t_ds.map(process_img, num_parallel_calls=AUTOTUNE)
    v_ds = v_ds.map(process_img, num_parallel_calls=AUTOTUNE)

    # 数据增强
    t_ds = t_ds.map(aug_img, num_parallel_calls=AUTOTUNE)
    v_ds = v_ds.map(aug_img, num_parallel_calls=AUTOTUNE)

    # 构建模型
    # 先用简单的keras
    model = tf.keras.Sequentail([
        # 卷积层
        layers.Conv2D(32, 3, activation='relu', input_shap=(400, 400, 3)),
        layers.MaxPooling2D(2),
        layers.Conv2D(64, 3, activation='relu'),
        layers.MaxPooling2D(2),
        layers.Conv2D(128, 3, activation='relu'),
        layers.MaxPooling2D(2),

        # 全连接层
        # 展开
        layers.Flatten(),
        # 全连接第一层
        layers.Dense(64, activation='relu'),
        # 丢弃神经元
        layers.DropOut(0.5),
        # 第二层,就是输出层
        layers.Dense(5)
    ])

    # 定义优化器
    opt = 'adam'

    # 定义损失函数，交叉熵损失
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logtis=True)

    # 编译模型
    model.complie(
        optimizer=opt,
        loss=loss_fn,
        metrics=['accuracy']
    )

    # 训练模型
    model.fit(t_ds, epoches=5)

    # 评估模型
    model.evaluate(v_ds, verbose=2)

    # 保存模型
    model.save('../face_catch.h5')


def process_img(img, label):
    img = tf.image.resize(400, 400)
    # 归一化
    img = img / 255.0

    # 标准化
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    img = (img - mean) / std

    return img, label


def aug_img(img, label):
    return tf.keras.Sequentail([
        tf.keras.layers.RandomFlip('horizontal'),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomContrast(0.2),
    ])
