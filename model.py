from torch import nn


class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linearLayerOne = nn.Linear(4096, 1024)
        self.linearLayerTwo = nn.Linear(1024, 128)
        self.linearLayerThree = nn.Linear(128, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.linearLayerOne(x))
        x = self.relu(self.linearLayerTwo(x))
        return self.linearLayerThree(x)

