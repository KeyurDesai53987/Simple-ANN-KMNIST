import torch.nn as nn

# Model Architecture
class SimpleANN(nn.Module):
    def __init__(self):
        super(SimpleANN, self).__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 10),
            nn.BatchNorm1d(10),
            nn.Softmax(dim=1)
        )
        self.init_weights()

    def init_weights(self):
        for module in self.model.modules():
            if isinstance(module, nn.Linear):
                nn.init.uniform_(module.weight, a=-0.1, b=0.1)
                nn.init.constant_(module.bias, 0.0)

    def forward(self, x):
        return self.model(x)