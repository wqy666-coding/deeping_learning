from keras import models
from keras import layers
from keras import optimizers
# 定义神经网络结构
model = models.Sequential()
model.add(layers.Dense(32,activation = 'relu',input_shape =(784,)))
model.add(layers.Dense(500,activation = 'sigmoid'))
model.add(layers.Dense(10,activation = 'softmax'))
# model.summary()

from keras import optimizers
model.compile(optimizer = optimizers.RMSprop(lr =0.001),
loss = 'mse',metrics = ['accuracy']) #MES（Mean Squared Error，均方误差）  

# 数据集的探索和初始化
from keras.datasets import mnist
import numpy as np
(train_images,train_labels),(test_images,test_labels) = mnist.load_data()

print(train_images.shape)


# print(type(train_images))
# print(type(train_images[1,1,1]))
#
# print(type(train_labels[1]))

print(train_images.shape)
train_images = train_images.reshape((60000,28*28)) #二维变一维
# print(train_images.shape)

train_images = train_images.astype("float32")/255 # 归一化
# print(test_images.shape)
test_images = test_images.reshape((10000,28*28))
# print(test_images.shape)
test_images = test_images.astype('float32')/255


from keras.utils import to_categorical #分类
# print(train_labels.shape)
# print(train_labels)
train_labels = to_categorical(train_labels) # 变成one-hot编码
# print(train_labels)

test_labels = to_categorical(test_labels)

# 模型训练
model.fit(train_images,train_labels,batch_size = 128,epochs=10)
# 预测
print(np.round(model.predict(test_images[0:5][:])).astype(int))

print(test_labels[0:5])
# 测试
loss, accuracy = model.evaluate(test_images, test_labels)
print(loss)
print(accuracy)



