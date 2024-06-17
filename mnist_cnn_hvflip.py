# packages
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict


# 1. Create Neural Network
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # kernel
        # 1 input image channel, 64 output channels, 3x3 square convolution, 1 stride, 1 padding
        self.conv1 = nn.Conv2d(1, 64, 3, 1, 1)
        self.norm1 = nn.BatchNorm2d(64)

        # 64 input channels, 128 output channels, 3x3 square convolution, 1 stride, 1 padding
        self.conv2 = nn.Conv2d(64, 128, 3, 1, 1)
        self.norm2 = nn.BatchNorm2d(128)
        
        # 128 input channels, 256 output channels, 3x3 square convolution, 1 stride, 1 padding
        self.conv3 = nn.Conv2d(128, 256, 3, 1, 1)
        self.norm3 = nn.BatchNorm2d(256)

        # 256 input channels, 256 output channels, 3x3 square convolution, 1 stride, 1 padding
        self.conv4 = nn.Conv2d(256, 256, 3, 1, 1)
        self.norm4 = nn.BatchNorm2d(256)

        # 256 input channels, 512 output channels, 3x3 square convolution, 1 stride, 1 padding
        self.conv5 = nn.Conv2d(256, 512, 3, 1, 1)
        self.norm5 = nn.BatchNorm2d(512)

        # 512 input channels, 512 output channels, 3x3 square convolution, 1 stride, 1 padding
        self.conv6 = nn.Conv2d(512, 512, 3, 1, 1)
        self.norm6 = nn.BatchNorm2d(512)

        # 512 input channels, 512 output channels, 3x3 square convolution, 1 stride, 1 padding
        self.conv7 = nn.Conv2d(512, 512, 3, 1, 1)
        self.norm7 = nn.BatchNorm2d(512)

        # 512 input channels, 512 output channels, 3x3 square convolution, 1 stride, 1 padding
        self.conv8 = nn.Conv2d(512, 512, 3, 1, 1)
        self.norm8 = nn.BatchNorm2d(512)
        
        self.fc1 = nn.Linear(512, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 10)

        self.drop1 = nn.Dropout(0.5)
        self.drop2 = nn.Dropout(0.5)

    def forward(self, input):
        # Convolution layer C1: 1 input image channel, 64 output channels,
        # 3x3 square convoluton, first batch-normalize then RELU, and
        # outputs a Tensor with size (N, 64, 32, 32) where N is the size of the batch
        c1 = F.relu(self.norm1(self.conv1(input)))

        # Subsampling layer S2: 2x2 grid, outputs a (N, 64, 16, 16) Tensor
        s2 = F.max_pool2d(c1, (2,2))

        # Convolution layer C3: 64 input channels, 128 output channels,
        # 3x3 square convolution, first batch-normalize then RELU, and
        # outputs a Tensor with size (N, 128, 16, 16)
        c3 = F.relu(self.norm2(self.conv2(s2)))

        # Subsampling layer S4: 2x2 grid, outputs a (N, 128, 8, 8) Tensor
        s4 = F.max_pool2d(c3, (2,2))

        # Convolution layer C5: 128 input channels, 256 output channels,
        # batch-normalize then RELU, and
        # outputs a Tensor with size (N, 256, 8, 8)
        c5 = F.relu(self.norm3(self.conv3(s4)))

        # Convolution layer C6: 256 input channels, 256 output channels,
        # batch-normalize, then RELU, and
        # outputs a Tensor with size (N, 256, 8, 8)
        c6 = F.relu(self.norm4(self.conv4(c5)))

        # Subsampling layer S7: 2x2 grid, outputs a (N, 256, 4, 4) Tensor
        s7 = F.max_pool2d(c6, (2,2))

        # Convolution layer C8: 256 input channels, 512 output channels,
        # batch-normalize, then RELU, and
        # outputs a Tensor with size (N, 512, 4, 4)
        c8 = F.relu(self.norm5(self.conv5(s7)))

        # Convolution layer C9: 512 input channels, 512 output channels,
        # batch-normalize, then RELU, and
        # outputs a Tensor with size (N, 512, 4, 4)
        c9 = F.relu(self.norm6(self.conv6(c8)))

        # Subsampling layer S10: 2x2 grid, outputs a (N, 512, 2, 2) Tensor
        s10 = F.max_pool2d(c9, (2,2))

        # Convolution layer C11: 512 input channels, 512 output channels,
        # batch-normalize, then RELU, and
        # outputs a Tensor with size (N, 512, 2, 2)
        c11 = F.relu(self.norm7(self.conv7(s10)))

        # Convolution layer C12: 512 input channels, 512 output channels,
        # batch-normalize, then RELU, and
        # outputs a Tensor with size (N, 512, 2, 2)
        c12 = F.relu(self.norm8(self.conv8(c11)))

        # Subsampling layer S13: 2x2 grid, outputs a (N, 512) Tensor
        s13 = F.max_pool2d(c12, (2,2))

        # Fully connected layer F14: (N, 512) Tensor input,
        # and outputs a (N, 4096) Tensor output, uses RELU
        x14 = torch.flatten(s13, 1)
        f14 = F.relu(self.fc1(x14))

        # Dropout layer D15: (N, 4096) Tensor input and output
        d15 = self.drop1(f14)

        # Fully connected layer F16: (N, 4096) Tensor input,
        # and outputs a (N, 4096) Tensor output, uses RELU
        f16 = F.relu(self.fc2(d15))
        
        # Dropout layer D17: (N, 4096) Tensor input and output
        d17 = self.drop2(f16)

        # Fully connected layer f18: (N, 4096) Tensor input,
        # and outputs a (N, 10) Tensor output
        output = self.fc3(d17)
        return output
        

# 2. Set parameters
batch_size_train = 256
batch_size_test = 64
learning_rate = 0.01
momentum = 0.9
epochs = 5
net = Net()
optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum)
loss_fn = nn.CrossEntropyLoss()

# 3. Import Data, Transform and Load
norm_resize_transformer = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
    transforms.Resize((32, 32))
])

training_data = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=norm_resize_transformer
)

test_data = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=norm_resize_transformer
)

train_loader = DataLoader(training_data, batch_size = batch_size_train)
test_loader = DataLoader(test_data, batch_size=batch_size_test)

hflip_test_data = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=transforms.Compose([norm_resize_transformer, transforms.RandomHorizontalFlip(p=1)])
)

vflip_test_data = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=transforms.Compose([norm_resize_transformer, transforms.RandomVerticalFlip(p=1)])
)

hflip_test_loader = DataLoader(hflip_test_data, batch_size=batch_size_test)

vflip_test_loader = DataLoader(vflip_test_data, batch_size=batch_size_test)


# 4. Define Training Function per Epoch
def train_one_epoch(epoch_idx):
    # per epoch, we train over sets of batches
    total_loss = 0.0
    total_correct = 0.0
    for i, data in enumerate(train_loader):
        inputs, labels = data

        optimizer.zero_grad()

        outputs = net(inputs)

        loss = loss_fn(outputs, labels)
        print("epoch: " + str(epoch_idx + 1) + "    batch index: " + str(i) + "    loss: " + str(loss.item()))
        loss.backward()

        optimizer.step()

        _, preds = torch.max(outputs.data, 1)
        total_correct += (preds == labels).float().sum()
        total_loss += loss
    # evaluate training loss and accuracy
    avg_loss = total_loss / (i + 1)
    avg_accuracy = total_correct / ((i + 1) * batch_size_train)
    return avg_loss, avg_accuracy



# 5. Define Training + Testing Function
def train_test():
    vflip_v_acc_list = [None] * epochs
    hflip_v_acc_list = [None] * epochs
    vflip_dict = defaultdict(int)
    hflip_dict = defaultdict(int)
    
    for epoch in range(epochs):
        print('EPOCH: ' + str(epoch + 1) + "  ############################")

        net.train(True)
        training_loss, training_accuracy = train_one_epoch(epoch)

        net.eval()
        hflip_total_correct = 0
        vflip_total_correct = 0
        with torch.no_grad():
            for i, data in enumerate(hflip_test_loader):
                inputs, labels = data
                outputs = net(inputs)
                _, preds = torch.max(outputs.data, 1)
                hflip_total_correct += (preds == labels).float().sum()
                for idx in range(len(preds)):
                    key = (int(preds[idx]), int(labels[idx]))
                    if preds[idx] == labels[idx]:
                        hflip_dict[key] += 1
                    else:
                        hflip_dict[key] -= 1
            hflip_test_accuracy = hflip_total_correct / ((i + 1) * batch_size_test)
            for i, data in enumerate(vflip_test_loader):
                inputs, labels = data
                outputs = net(inputs)
                _, preds = torch.max(outputs.data, 1)
                vflip_total_correct += (preds == labels).float().sum()
                for idx in range(len(preds)):
                    key = (int(preds[idx]), int(labels[idx]))
                    if preds[idx] == labels[idx]:
                        vflip_dict[key] += 1
                    else:
                        vflip_dict[key] -= 1
            vflip_test_accuracy = vflip_total_correct / ((i + 1) * batch_size_test)          


        print("For Epoch {}, we have {} hflip test accuracy and {} vflip test accuracy."
              .format(epoch + 1, hflip_test_accuracy, vflip_test_accuracy))
        vflip_v_acc_list[epoch] = vflip_test_accuracy
        hflip_v_acc_list[epoch] = hflip_test_accuracy
    return hflip_v_acc_list, vflip_v_acc_list, hflip_dict, vflip_dict


# train, test, graph results
hflip_acc, vflip_acc, hflip_dict, vflip_dict = train_test()
x = np.arange(1, epochs+1)
plt.plot(x, hflip_acc, color='red', label='h-flip')
plt.plot(x, vflip_acc, color='blue', label='v-flip')
plt.legend()
plt.title('Test Accuracy (Flipped Images) vs Epoch')
plt.ylim([0, 1])
plt.xticks(x)
plt.savefig("flip_results.png")

