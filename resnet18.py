import torch
import numpy as np
import torchvision
from torchvision import datasets, transforms, models
from torch.utils import data
import matplotlib.pyplot as plt
import os
import time

###########################
batch_size = 4
num_epochs = 1
learning_rate = 0.001
###########################

FILE = "saves/resnet18.pth"
device = torch.device("mps")

def train(train_loader):
    # resnet with last fully connected layer that outputs 2 values
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = torch.nn.Linear(model.fc.in_features, 2)
    model = model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    start = time.time()

    n_total_steps = len(train_loader)
    for epoch in range(num_epochs):
        i = 0
        for (images, labels) in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            i += 1
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if(i % 1 == 0):
                print(f'epoch {epoch+1} / {num_epochs}, step {i}/{n_total_steps}, loss = {loss.item():.4f}')


    print(f'Time for learning: {float(time.time() - start) /60:.1f}m')
    torch.save(model.state_dict(), FILE)
    print(f'Model saved in {FILE}')

def test(test_loader, model):
    n_correct_preds = 0
    n_samples = 0
    for (images, labels) in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, pred = torch.max(outputs, 1)
        n_samples += 1
        for i in range(batch_size):
            if(pred[i] == labels[i]):
                n_correct_preds += 1
    print(f'Accuracy: {100*n_correct_preds/n_samples:.1f}%')
        

if __name__ == "__main__":
    
    transform = transforms.Compose([
        transforms.Resize((255, 255)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = datasets.ImageFolder("./images/test/", transform=transform)
    test_dataset = datasets.ImageFolder("./images/finetune/", transform=transform)
    train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    dataset_sizes = {'train': len(train_dataset), 'test': len(test_dataset)}
    class_names = train_dataset.classes

    print(f'Classes: {class_names}')
    print(f'Datasets sizes: {dataset_sizes}')
    
    train(train_loader)
    
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = torch.nn.Linear(model.fc.in_features, 2)
    model = model.to(device)
    
    model.load_state_dict(torch.load(FILE))
    
    test(test_loader, model)
    
