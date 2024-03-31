from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('logs')
for i in range(100):
    writer.add_scalar("y=3x",3*i,i)
writer.close()

# from PIL import Image
# from 获取图片数据 import MyData
# root_path = '../lianxi_data/train/'
# label  = 'ants_image'
# ants_data = MyData(root_path,label)
# img_path,img_cv, label = ants_data.__getitem__(0)
# img = Image.open(img_path)
# print(type(img))
import numpy as np


#
