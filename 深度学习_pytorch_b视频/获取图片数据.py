import cv2 as cv

# 读取图片
# img = cv.imread("../hymenoptera_data/train/ants/0013035.jpg")
# 现在图片
# cv.imshow('img1',img)
# # 设置一个等待
# cv.waitKey(0)
# # 摧毁所有窗口
# cv.destroyAllWindows()

import os
# 把图片名称放入列表中
# dir_path = '../hymenoptera_data/train/'ants''
# img_path_list  = os.listdir(dir_path)
# for img in img_path_list:
#     img1 = cv.imread("../hymenoptera_data/train/ants/"+img)
#     cv.imshow('img1', img1)
#     # 设置一个等待
#     cv.waitKey(0)
#     # 摧毁所有窗口
#     cv.destroyAllWindows()

from torch.utils.data import Dataset

class MyData(Dataset):
    def __init__(self,root_dir,label_dir):
        self.root = root_dir
        self.label = label_dir
        self.path = os.path.join(self.root,self.label)
        self.img_path = os.listdir(self.path)
    def __getitem__(self, idx):
        img_name = self.img_path[idx]
        img_item_path = os.path.join(self.root,self.label,img_name)
        img = cv.imread(img_item_path)
        label = self.label
        return img,label
    def __len__(self):
        return len(self.img_path)
    def show(self,num):
        img,label = self.__getitem__(num)
        cv.imshow('img',img)
        cv.waitKey(0)
        cv.destroyAllWindows()
if __name__ =='__main__':
    # 实例化对象
    ants_data = MyData('../hymenoptera_data/train','ants')
    bees_data = MyData('../hymenoptera_data/train','bees')
    # 获取第一张图片
    # data.__getitem__(0)
