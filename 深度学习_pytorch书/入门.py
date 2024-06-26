import torch
# 创建张量
x = torch.arange(12)
# print(x) #tensor([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11])

# 查看张量的形状
# print(x.shape) # torch.Size([12])

# 查看张量中元素的总数

# print(x.numel()) # 12

# 改变张量的形状
y = x.reshape(3,4)
# print(y)
#tensor([[ 0,  1,  2,  3],
#        [ 4,  5,  6,  7],
#        [ 8,  9, 10, 11]])

# 设置全为0的张量 或者 全为1的张量

a = torch.zeros((2,3,4))
# print(a)

"""
tensor([[[0., 0., 0., 0.],
         [0., 0., 0., 0.],
         [0., 0., 0., 0.]],

        [[0., 0., 0., 0.],
         [0., 0., 0., 0.],
         [0., 0., 0., 0.]]])
"""
# 设置全为1的张量
b = torch.ones((2,3,4))
# print(b)

"""
tensor([[[1., 1., 1., 1.],
         [1., 1., 1., 1.],
         [1., 1., 1., 1.]],

        [[1., 1., 1., 1.],
         [1., 1., 1., 1.],
         [1., 1., 1., 1.]]])


"""

# 将大小为1的张量转化为python标量
a = torch.tensor([3.5])
print(a) # tensor([3.5000])
print(a.item()) #3.5
print(int(a)) #3