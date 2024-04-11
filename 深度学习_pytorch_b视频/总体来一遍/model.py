import torch
from torch import nn


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

if __name__ == "__main__":
    # input = torch.ones((64,3,32,32))
    tutui = Tutui()
    # output = tutui(input)
    # print(output.shape)
    torch.save(tutui, "../tutui.pth")
