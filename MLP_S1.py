from torch.utils.data import DataLoader
import torch
import torchvision
from torchvision import datasets, transforms
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

def mlp_apply(model,ind):
    mnist_test = datasets.FashionMNIST('D:\M.Sc. Artificial Intelligence\Semester 1\Artificial Neural Networks and Cognitive Models',train = False, download = True,transform=transforms.ToTensor())
    true = 0
    for i in range(10):
        n = ind[i]
        img = mnist_test[i][0]
        label = mnist_test[i][1]
        label_name =["T-shirt/top","Trouser","Pullover","Dress","Coat","Sandal","Shirt","Sneaker","Bag","Ankle boot"]
        pred = model(img)
        pred = torch.argmax(pred, dim = 1)
        plt.imshow(img.squeeze(), cmap = 'gray')
        plt.show()
        print("True Label: ",label, '(',label_name[label],')')
        print("Predicted Label: ",int(pred), '(',label_name[pred],')')
        if(label == pred):
            true +=1
    print(true,"/","10") 
def accuracy(model,indices):
    mnist_test = datasets.FashionMNIST('D:\M.Sc. Artificial Intelligence\Semester 1\Artificial Neural Networks and Cognitive Models',train = False, download = True,transform=transforms.ToTensor())
    p=0
    for i in (indices):
        pred = model(mnist_test[i][0])
        #print (torch.argmax(pred, dim =1),mnist_test[i][1])
        d = (torch.argmax(pred, dim =1)==mnist_test[i][1])
        if(d):
            p+=1
    #print(p)       
    return((p/len(indices))*100)    
def seq(h,indim=28*28):
    seql=[nn.Flatten()]
    for l in range(len(h)):
        seql.append(nn.Linear(indim,h[l],bias = True))
        seql.append(nn.ReLU())
        indim = h[l]
    seql.append(nn.Linear(h[-1],10,bias = True))    
    seql.append(nn.ReLU())    
    return seql 

def mlp_train(hidden_dims, epochs, batch_size, learning_rate, cuda, plots):
   
    mnist_train = torchvision.datasets.FashionMNIST(root= './data/FashionMNIST',train = True,download = True,transform = transforms.Compose([transforms.ToTensor()]))
    mnist_test = torchvision.datasets.FashionMNIST(root= './data/FashionMNIST',train = False,download = True,transform = transforms.Compose([transforms.ToTensor()]))
    dl_train = DataLoader(mnist_train, batch_size, shuffle = True)
    dl_test = DataLoader(mnist_test, batch_size)
    model = nn.Sequential(*seq(hidden_dims))
    loss = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr = learning_rate)
    losses = []
    losses_test = []
    for epoch in range (epochs):
        print("Epoch: ",epoch+1)
        for batch in dl_train:
            x_batch, y_batch = batch[0],batch[1]
            if (cuda):
                x_batch = x_batch.to('cuda')
                y_batch = y_batch.to('cuda')
                model = model.to('cuda')
            preds = model(x_batch)
            l= loss(preds, y_batch)
            optimizer.zero_grad()
            l.backward(retain_graph=True)
            optimizer.step()
            losses.append(l.item())
        with torch.no_grad():
            for batch in dl_test:
                l = 0
                x_batch, y_batch = batch[0],batch[1]
                if (cuda):
                    x_batch = x_batch.to('cuda')
                    y_batch = y_batch.to('cuda')
                    model = model.to('cuda')
                preds = model(x_batch)
                l += loss(preds, y_batch)
            losses_test.append(l.item())
    for name, param in model.named_parameters():
        print("Executed Device: ",param.device)
        break       
    model.to('cpu')
    ltuple = (losses,losses_test)
    if (plots):
        plt.title('Training loss')
        plt.plot(losses)
        plt.show()
        plt.title('Test loss')
        plt.plot(losses_test)
              
    return model,ltuple     
   