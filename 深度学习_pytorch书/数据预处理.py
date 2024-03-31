'''


import os

# 创建一个data文件
os.makedirs(os.path.join("..",'data'),exist_ok=True)
#
# data_file = os.path.join("..","data",'house_tiny.csv')
with open(data_file,'w') as f:
    f.write('NumRooms,Alley,Price\n')  # 列名
    f.write('NA,Pave,127500\n')  # 每行表示一个数据样本
    f.write('2,NA,106000\n')
    f.write('4,NA,178100\n')
    f.write('NA,NA,140000\n')
'''

# 加载csv文件

import pandas as pd
data = pd.read_csv("../data/house_tiny.csv")
# print(data)

# 处理缺失值

#插值法:用一个替代值弥补缺失值
#通过位置索引iloc，我们将data分成inputs和outputs，其中前者为data的前两列，而后者为data的最后一列。
# 对于inputs中缺少的数值，我们用同一列的均值替换“NaN”项。
inputs,outputs = data.iloc[:,0:1],data.iloc[:,2]
inputs = inputs.fillna(inputs.mean())
inputs['Alley'] = ['Pave','NA','NA','NA']
# print(inputs)

# 使用get_dummies进行独热编码
inputs = pd.get_dummies(inputs, dummy_na=True)
# print(inputs)

# 将True替换为1，False替换为0
inputs = inputs.astype(int)
# print(inputs)

# 删除名为'Alley_NA'的列
inputs = inputs.drop('Alley_nan', axis=1)
# print(inputs)

# 交换 Alley_NA  Alley_Pave 两列
inputs = inputs[['NumRooms','Alley_Pave','Alley_NA']]
# 修改列名
inputs = inputs.rename(columns= {'Alley_NA':'Alley_nan'})
# print(inputs)

# 现在inputs和outputs中的所有条目都是数值类型，它们可以转换为张量格式。
import torch
# 将inputs DataFrame 转换为 NumPy 数组，并将其转换为 PyTorch 张量（Tensor）X。
X = torch.tensor(inputs.to_numpy(dtype=float))
'''
tensor([[3., 1., 0.],
        [2., 0., 1.],
        [4., 0., 1.],
        [3., 0., 1.]], dtype=torch.float64)
'''
Y = torch.tensor(outputs.to_numpy(dtype=float))
#tensor([127500., 106000., 178100., 140000.], dtype=torch.float64)
# print(Y)

################################### 线性代数 #######################################
# 仅包含一个数值被称为标量
a = torch.tensor(20)
# print(a)#tensor(20)
# print(a.shape) #torch.Size([])

################################### 微积分 #######################################
import numpy as np
from matplotlib_inline import backend_inline
from d2l import torch as d2l

def use_svg_display(): #@save
    """使用svg格式显示绘图"""
    backend_inline.set_matplotlib_formats('svg')

def set_figsize(figsize=(3.5, 2.5)): #@save
    """设置matplotlib的图表大小"""
    use_svg_display()
    d2l.plt.rcParams['figure.figsize'] = figsize
#@save
def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    """设置matplotlib的轴"""
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_xscale(xscale)
    axes.set_yscale(yscale)
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)
    if legend:
        axes.legend(legend) #添加图例
    axes.grid()
#@save
def plot(X, Y=None, xlabel=None, ylabel=None, legend=None, xlim=None,
ylim=None, xscale='linear', yscale='linear',
fmts=('-', 'm--', 'g-.', 'r:'), figsize=(3.5, 2.5), axes=None):
    """绘制数据点"""
    if legend is None:
        legend = []
    set_figsize(figsize)
    axes = axes if axes else d2l.plt.gca() #获取当前的坐标轴（Current Axes）

    # 如果X有一个轴，输出True
    def has_one_axis(X):

        return (hasattr(X, "ndim") and X.ndim == 1 or isinstance(X, list)
            and not hasattr(X[0], "__len__"))
    if has_one_axis(X):
        X = [X]
    if Y is None:
        X, Y = [[]] * len(X), X
    elif has_one_axis(Y):
        Y = [Y]
    if len(X) != len(Y):
        X = X * len(Y)
    axes.cla()# 这个方法通常在需要重新绘制新的图形之前使用，以确保不会在同一轴域上叠加之前的绘图内容。
    for x, y, fmt in zip(X, Y, fmts):
        if len(x):
            axes.plot(x, y, fmt)
        else:
            axes.plot(y, fmt)
    set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend)

def f(x):
    return 3 * x ** 2 - 4 * x
x = np.arange(0, 3, 0.1)
plot(x, [f(x), 2 * x - 3], 'x', 'f(x)', legend=['f(x)', 'Tangent line (x=1)'])
d2l.plt.show()

