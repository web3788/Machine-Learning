import torchvision
from torch import nn
import torch
from torch.utils.data import DataLoader
from matplotlib import font_manager


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_losses = []
test_losses = []

train_dataset = torchvision.datasets.CIFAR10(root='dataset', train=True, transform=torchvision.transforms.Compose([
    torchvision.transforms.RandomCrop(size=(32, 32), padding=4),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
]), download=True)

test_dataset = torchvision.datasets.CIFAR10(root='dataset', train=False, transform=torchvision.transforms.Compose([
    torchvision.transforms.RandomCrop(size=(32, 32), padding=4),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
]), download=True)

train_data_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True, drop_last=True)
test_data_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=True, drop_last=True)

train_data_size = len(train_dataset)
test_data_size = len(test_dataset)

print(f'训练集的大小为{train_data_size}')
print(f'测试集的大小为{test_data_size}')


class MyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(num_features=32),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(num_features=64),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
        )

        self.fc = nn.Sequential(
            nn.Flatten(),

            nn.Linear(128 * 4 * 4, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(),

            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(),

            nn.Linear(256, 10)
        )

    def forward(self, x):
        return self.fc(self.main(x))


mynet = MyNet()
mynet = mynet.to(device)
print(mynet)

loss_fn = nn.CrossEntropyLoss()
loss_fn = loss_fn.to(device)

learning_rate = 1e-3
optimizer = torch.optim.Adam(mynet.parameters(), lr=learning_rate)

train_step = 0
test_step = 0

epochs = 5

if __name__ == '__main__':
    for epoch in range(epochs):
        print(f'----------第{epoch+1}轮训练开始----------')
        mynet.train()
        for batch_idx, (images, targets) in enumerate(train_data_loader):

            images = images.to(device)
            targets = targets.to(device)

            outputs = mynet(images)
            loss = loss_fn(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_step += 1
            if train_step % 100 == 0:
                print(f'训练第{train_step}次，loss={loss}')
                train_losses.append(loss)

        mynet.eval()
        accuracy_total = 0
        with torch.no_grad():
            for batch_idx, (images, targets) in enumerate(test_data_loader):

                images = images.to(device)
                targets = targets.to(device)

                outputs = mynet(images)
                loss = loss_fn(outputs, targets)

                accuracy = (outputs.argmax(axis=1) == targets).sum()
                accuracy_total += accuracy.item()
                test_step += 1

                if test_step % 100 == 0:
                    test_losses.append(loss)

            print(f'第{epoch+1}轮训练结束，准确率{accuracy_total/test_data_size}')
            torch.save(mynet, f'CIAFR_10_{epoch+1}_acc_{accuracy_total/test_data_size}.pth')