import torch
import torch.nn as nn

class OurModule(nn.Module):
    def __init__(self, num_inputs, num_classes, dropout_prob=0.3):
        # Return temporary object to reference parent object with  call to super()
        # OurModule == Subclass; nn.Module == Superclass
        super(OurModule, self).__init__()
        self.pipe = nn.Sequential(
            nn.Linear(num_inputs, 5),
            nn.ReLU(),
            nn.Linear(5, 20),
            nn.Dropout(p=dropout_prob),
            nn.Softmax(dim=1)
        )

    # Run model forward.
    # Forward is specified in as a __call__ method in nn.Module. Hence, calling the class will result in a default forward call.
    # Must specify forward explicitly to referencing our network implimentation 'OurModule'
    # in order to override default __call__ to forward with nn.Module.
    def forward(self, x):
        return self.pipe(x)


if __name__ == "__main__":
    net = OurModule(num_inputs=2, num_classes=3)
    print(net)

    v = torch.FloatTensor([[2, 3]])
    out = net(v)
    print(out)

    print("Cudas availability is %s" % torch.cuda.is_available())
    if torch.cuda.is_available():
        print("Data from cuda: %s" % out.to("cuda"))
