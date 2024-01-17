import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import  datasets
import torchvision.transforms as transforms

batch_size = 100

#CIFAR-10 数据集分集
train_dataset = datasets.CIFAR10(root='./data', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = datasets.CIFAR10(root='./data', train=False, transform=transforms.ToTensor(), download=True)


#加载数据
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

train_data_size = len(train_dataset)
test_data_size = len(test_dataset)


class neural_net(nn.Module):
    def __init__(self,input_num,hidden1_size,hidden2_size,out_put):
        super(neural_net,self).__init__()
        self.layer1 = nn.Linear(input_num,hidden1_size)
        self.layer2 = nn.Linear(hidden1_size,hidden2_size)   
        self.layer3 = nn.Linear(hidden2_size,out_put)

    def forward(self,x):
       x = self.layer1(x)
       x = torch.relu(x)
       x = self.layer2(x)
       x = torch.relu(x)
       x = self.layer3(x)
       return x
   

input_size = 32*32*3
hidden1_size = 500
hidden2_size = 200
class_conut = 100


net = neural_net(input_size,hidden1_size,hidden2_size,class_conut)
print(net)

learning_rate = 0.2
num= 10
celoss = nn.CrossEntropyLoss()
optimizer =  torch.optim.SGD(net.parameters(),lr=learning_rate)

def train(num):
    for epoch in range(num):
        print('epoch is ',epoch)
        for i,(images,labels) in enumerate(train_loader):
            images = Variable(images.view(images.size(0),-1))
            labels = Variable(labels)
            
            output = net(images)
            loss = celoss(output,labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if i%100 == 0:
                print('\t loss = %.6f' % loss.item()) 
    print('train finish')


train(num)



def test():
    total = 0 
    correct  = 0
    for images,labels in test_loader:
       images = Variable(images.view(images.size(0),-1))
       outputs = net(images)
       _,predict = torch.max(outputs.data,1)
       total += labels.size(0)
       correct += (predict == labels).sum()
    print('Accuracy = %.2f ' %(100*(correct/total)))

test()    