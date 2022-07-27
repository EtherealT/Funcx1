

def train():
    import torch
    import torchvision
    import torch.nn as nn
    import torch.optim as optim
    import torchvision.transforms as transforms
    import torch.nn as nn
    import torch.nn.functional as F

    class LeNet(nn.Module):
        def __init__(self):
            super(LeNet, self).__init__()
            self.conv1 = nn.Conv2d(3, 16, 5)
            self.pool1 = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(16, 32, 5)
            self.pool2 = nn.MaxPool2d(2, 2)
            self.fc1 = nn.Linear(32 * 5 * 5, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 10)

        def forward(self, x):
            x = F.relu(self.conv1(x))  # input(3, 32, 32) output(16, 28, 28)
            x = self.pool1(x)  # output(16, 14, 14)
            x = F.relu(self.conv2(x))  # output(32, 10, 10)
            x = self.pool2(x)  # output(32, 5, 5)
            x = x.view(-1, 32 * 5 * 5)  # output(32*5*5)
            x = F.relu(self.fc1(x))  # output(120)
            x = F.relu(self.fc2(x))  # output(84)
            x = self.fc3(x)  # output(10)
            return x
    transform = transforms.Compose( [transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # 50000张训练图片
    # 第一次使用时要将download设置为True才会自动去下载数据集
    train_set = torchvision.datasets.CIFAR10(root='./data', train=True,
                                             download=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=36,
                                               shuffle=False, num_workers=0)

    # 10000张验证图片
    # 第一次使用时要将download设置为True才会自动去下载数据集
    val_set = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=False, transform=transform)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=5000,
                                             shuffle=False, num_workers=0)
    val_data_iter = iter(val_loader)
    val_image, val_label = val_data_iter.next()
    net = LeNet()
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    for epoch in range(5):  # loop over the dataset multiple times

        running_loss = 0.0
        for step, data in enumerate(train_loader, start=0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = net(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if step % 500 == 499:  # print every 500 mini-batches
                with torch.no_grad():
                    outputs = net(val_image)  # [batch, 10]
                    predict_y = torch.max(outputs, dim=1)[1]
                    accuracy = torch.eq(predict_y, val_label).sum().item() / val_label.size(0)

                    print('[%d, %5d] train_loss: %.3f  test_accuracy: %.3f' %
                          (epoch + 1, step + 1, running_loss / 500, accuracy))
                    running_loss = 0.0

    print('Finished Training')
    save_path = './Lenet.pth'
    torch.save(net.state_dict(), save_path)
    import os
    loc=os.getcwd()

    return loc

if __name__ == '__main__':
    train()