from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch
from torch import nn
from torch import optim
import torchvision
import time
# 获取训练集和测试集
trainset = torchvision.datasets.CIFAR10("D:/learn/学习资料/深度学习/深度学习_pytorch_b视频/dataset",train=True,transform=torchvision.transforms.ToTensor())
testset = torchvision.datasets.CIFAR10("D:/learn/学习资料/深度学习/深度学习_pytorch_b视频/dataset",train=False,transform=torchvision.transforms.ToTensor())

# 打包（每64一组）
trainloader = DataLoader(trainset,batch_size=64)
testloader = DataLoader(testset,batch_size=64)

# 训练集和测试集的长度
trainset_size = len(trainset)
testset_size = len(testset)
print("训练集的个数",trainset_size)
print("测试集的个数",testset_size)

class Tutui(nn.Module):
    def __init__(self):
        super(Tutui,self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3,32,5,padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(32,32,5,padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(32,64,5,padding=2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(1024,64),
            nn.Linear(64,10)
        )
    def forward(self,x):
        x = self.model(x)
        return x

# 构建网络
model = Tutui()

# 使用gpu
if torch.cuda.is_available():
    model = model.cuda()

# 损失函数
loss = nn.CrossEntropyLoss()
# 使用gpu
loss = loss.cuda()
# 优化器
learning_rate = 1e-2
optimter = optim.SGD(model.parameters(),lr=learning_rate)

#训练的轮数
epcho = 10
# 训练次数
train_step = 0
# 测试次数
test_step = 0
# 创建日志
writer  = SummaryWriter("train")
# 整体测试的准确率
total_accuracy = 0
# 训练开始时间
start_time = time.time()
# 开始训练加测试
for i in range(epcho):
    print("------------第{}轮训练开始--------------".format(i+1))
    # 训练步骤开始
    model.train()

    for data in trainloader:
        imgs,targets = data
        # 使用gpu
        imgs = imgs.cuda()
        targets = targets.cuda()
        output = model(imgs)
        result_loss = loss(output,targets)
        # # 把可调节参数的梯度调为0
        optimter.zero_grad()
        # 获取每个结点梯度的参数
        result_loss.backward()
        # 优化器调优
        optimter.step()
        train_step+=1
        if train_step%100==0:
            end_time = time.time()
            print("训练100次时，花费的时间为:{}".format(end_time-start_time))
            print("训练次数：{}，Loss:{}".format(train_step,result_loss.item()))
            writer.add_scalar("train_loss",result_loss.item(),train_step)
    with torch.no_grad():
        runing_loss = 0
        # 测试步骤开始
        model.eval()
        for test in testloader:
            imgs,targets = test
            # 使用gpu
            imgs = imgs.cuda()
            targets = targets.cuda()
            output = model(imgs)
            result_loss = loss(output,targets)
            runing_loss+=result_loss.item()
            test_step+=1
            accuracy = (output.argmax(1) == targets).sum()
            total_accuracy +=accuracy
    print("整体测试集上的损失Loss:{}".format(runing_loss))
    print("整体测试集上的正确率:{}".format(total_accuracy/testset_size))

    writer.add_scalar("test_loss", runing_loss, test_step)

    writer.add_scalar("test_accuracy", total_accuracy/testset_size, test_step)
    # 模型保存
    torch.save(model,"model_{}.pth".format(i))
    print("模型已保存")
writer.close()



