import torch.nn as nn

class RIID_NN(nn.Module):
    def __init__(self):
        super(RIID_NN, self).__init__()
        self.l1 = nn.Linear(in_features=4, out_features=10)
        self.l2 = nn.Linear(in_features=10, out_features=45)
        self.l3 = nn.Linear(in_features=45, out_features=30)
        self.l4 = nn.Linear(in_features=30, out_features=1)
        self.layer_act = nn.ReLU()
        self.act = nn.Sigmoid()
        
        self.apply(self.init_weights)
    
    def forward(self, input):
        out = self.layer_act(self.l1(input))
        out = self.layer_act(self.l2(out))
        out = self.layer_act(self.l3(out))
        out = self.act(self.l4(out))
        return out
    
    def init_weights(self, x):
        if isinstance(x, nn.Linear):
            nn.init.kaiming_normal_(x.weight.data)
            nn.init.constant_(x.bias.data, 0)
