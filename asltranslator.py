import numpy as np 
import pandas as pd # data processing
import os, sys
import time
import matplotlib.pyplot as plt

import tensorflow as tf
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
import torch.utils.data as data

import torchvision
import torchvision.transforms as transforms
import torchvision.models as models


# Presetup before generating and training NN
from google.colab import drive
drive.mount('/content/drive')

#set device so that the model and data will run on the correct graphics device
dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(dev)


class Net(nn.Module):
    def __init__(self):
        ''' init the nerual network '''
        super(Net, self).__init__()
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 8, 3),
            nn.BatchNorm2d(8),
            nn.AvgPool2d(2),
            nn.ReLU()
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(8, 16, 3),
            nn.BatchNorm2d(16),
            nn.AvgPool2d(2),
            nn.ReLU()
        )
        
        self.fc1 = nn.Linear(16*5*5, 100)
        self.fc2 = nn.Linear(100, 64)
        self.leak = nn.LeakyReLU()
        self.drop = nn.Dropout(p=0.3)
        self.fc3 = nn.Linear(64, 25)
        

    def forward(self, x):
        '''defines the forward prop algorithm'''
        #apply Convolution on the relu'd results from the convolution layers
        x = self.layer1(x)
        x = self.layer2(x)

        x = x.view((x.shape[0], -1))

        #run the fcs
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.leak(x)
        x = self.drop(x)

        #fc4 will give us the final **24** layers
        x = self.fc3(x)

        return x

net = Net()
#switch the nn to run on gpu if available
net = net.to(dev)
print(net)

#Define the Loss Function and Optimizer
crit = nn.CrossEntropyLoss()

optimizer = optim.Adam(net.parameters(), lr=0.001)


for dirname, _, filenames in os.walk('/content/drive/MyDrive/Colab Notebooks/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))



train = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/kaggle/input/sign_mnist_train.csv')
test = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/kaggle/input/sign_mnist_test.csv')
#Put in J as a _ to maintain 1to1 matching in nn
labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', '_', 'K', 'L', 'M', 
          'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']

#Create Class for Dataset to init and transform datasets

class SignLanguageDataset(data.Dataset):

    def __init__(self, df, transform=None):
        '''init dataset'''
        self.df = df
        self.transform = transform

    def __len__(self):
        '''init return length of dataset'''
        return self.df.shape[0]

    def __getitem__(self, index):
        '''define label and transform image based on index given'''
        label = self.df.iloc[index, 0]

        img = self.df.iloc[index, 1:].values.reshape(28, 28)
        img = torch.Tensor(img).unsqueeze(0)
        
        if self.transform is not None:
            img = self.transform(img)

        return img, label



#method for showing images using mathplt
def show_img(img, label):
    img = img.squeeze()
    img = img*40. + 159.
    imgnp = img.detach().numpy()
    plt.imshow(img, interpolation='bicubic')
    print(label)


train_transform = transforms.Compose([
    #randomly flip/rotate images to better train nn
    transforms.RandomHorizontalFlip(p=0.3),
    transforms.RandomApply([transforms.RandomRotation(degrees=(-180,180))],
                           p = 0.2)
])


train_dataset = SignLanguageDataset(train, transform=train_transform)
test_dataset = SignLanguageDataset(test)

train_loader = data.DataLoader(train_dataset, batch_size=200, shuffle=True,
                               num_workers=2)
test_loader = data.DataLoader(test_dataset, batch_size=200, shuffle=True,
                              num_workers=2)



trainiter = iter(train_loader)
img, label = next(trainiter)
print(img.shape)


show_img(img[10], label[10])

#Define Training and Eval
#Now that we have defined a simple conv2d nn and init all our training data, 
# we can start work on defining our training and evaluation algorithms
def eval_net(model, crit, test_loader):

    #this will switch the net to run on gpu if supported and selected
    #if there is no device chosen, default will be cpu
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    model = model.to(device)

    model = model.eval()

    running_loss = 0.0
    num_correct = 0.0
    num_total = 0.0

    for batch, labels in test_loader:

        batch = batch.to(device)
        labels = labels.to(device)

        out = model(batch)
        pred_labels = out.argmax(dim=1)
        num_correct += float((pred_labels == labels).sum())

        loss = crit(out, labels)
        running_loss += loss.data.cpu()

        num_total += labels.shape[0]

    mean_loss = running_loss / num_total
    accuracy = num_correct / num_total

    return mean_loss, accuracy




def train_model(n_epochs, model, optimizer, crit, train_loader,
                test_loader):
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    
    model = model.train()
    
    train_loss, train_acc = [], []
    test_loss, test_acc = [], []
    
    for epoch in range(n_epochs):
        
        t0 = time.perf_counter()
        
        running_loss = 0.
        num_correct = 0.
        num_total = 0.
        
        for batch, labels in train_loader:
            batch = batch.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            
            out = model(batch)
            
            pred_labels = out.argmax(dim=1)
            num_correct += float((pred_labels == labels).sum())
            num_total += labels.shape[0]
            
            loss = crit(out, labels)
            running_loss += loss
            loss.backward()
            optimizer.step()
        
        epoch_loss = running_loss / num_total
        epoch_acc = num_correct / num_total
        
        train_loss.append(epoch_loss.data.cpu())
        train_acc.append(epoch_acc)
        
        t_loss, t_acc = eval_net(model, crit, test_loader)
        
        test_loss.append(t_loss.data.cpu())
        test_acc.append(t_acc)
        
        t1 = time.perf_counter()
        
        delta_t = t1 - t0
        print(f"EPOCH {epoch+1} ({round(delta_t, 4)} s.): train loss - {epoch_loss}, train accuracy - {epoch_acc}; test loss - {t_loss}, test accuracy - {t_acc}")
    
    
    return model, train_loss, train_acc, test_loss, test_acc        
    


#Run the train_model func

net, train_loss, train_acc, test_loss, test_acc = train_model(20, net, optimizer, crit,
                                                     train_loader, test_loader)


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,6))
ax1.plot(train_loss)
ax1.plot(test_loss)
ax1.legend(['train', 'test'])
ax1.set_title('Loss')
ax2.plot(train_acc)
ax2.plot(test_acc)
ax2.legend(['train', 'test'])
ax2.set_title('Accuracy')

print("Accuracy: %.3f, %.3f" % (train_acc[19], test_acc[19]))


#Save Model

state = net.state_dict()

torch.save(state, './sign_lang_net.pth')


#Check Model
testiter = iter(test_loader)
img, label = next(testiter)
net = net.cpu()

image = img[0]
lab = label[0]

batch = train_transform(image).unsqueeze(0)
# if you want to run a single img through the nn do this
pred = net(batch).squeeze(0).softmax(0)
class_id = pred.argmax().item()
#class_id will give the most likely index.
#Ex: if class_id is 7 -> that means the expected value would be labels[7] or H

_, indicies = torch.sort(pred, descending=True)
    
for ind in indicies[:5]:
    i = ind.item()
    print(f"{labels[i]}: {100.*pred[i]:.2f}%")

print(f"Correct: {labels[lab]}")

show_img(batch, lab)



