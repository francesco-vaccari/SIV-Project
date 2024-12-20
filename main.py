import torch
import numpy as np
from torch.utils import data
from torch.optim import lr_scheduler
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
import time

PYTORCH_ENABLE_MPS_FALLBACK=1

device = torch.device("mps")

##################
batch_size = 4
num_epochs_cl = 1
max_worse_acc_finetuner = 10 # dopo n epochs in cui la eval loss aumenta interrompo finetuning
temperature = 0.07
learning_rate_cl = 0.0001 # fino a 0.000001
learning_rate_ft = 0.00001
FILECL = "cl.pth"
FILEFT = "ft.pth"
resize_x, resize_y = 256, 256 # dimension of input images
random_crop_size = (120, 120) # size of the cropped image for augmentations
##################


class ProjectionHead(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ProjectionHead, self).__init__()
        self.l1 = torch.nn.Linear(input_size, hidden_size)
        self.relu = torch.nn.ReLU()
        self.l2 = torch.nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        return out


class FineTuneNN(torch.nn.Module):
    def __init__(self):
        super(FineTuneNN, self).__init__()
        self.l1 = torch.nn.Linear(2048, 256)
        self.relu = torch.nn.ReLU()
        self.l2 = torch.nn.Linear(256, 2)
    
    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        return out


def random_crop(image, crop_size):
    (crop_height, crop_width) = crop_size
    x = np.random.randint(0, image.shape[1]- crop_height)
    y = np.random.randint(0, image.shape[2] - crop_width)
    crop = torch.zeros(3, crop_height, crop_width).to(device)
    for index_channel, channel in enumerate(image):
        for i in range(crop_height):
            for j in range(crop_width):
                crop[index_channel][i][j] = channel[x+i][y+j]
    return crop


# a differenza di quella implementata da SimCLR, evito di contrastare samples positivi al denominatore
def NT_Xent_loss(embeddings, index_positive1, index_positive2):
    positive1 = embeddings[index_positive1]
    positive2 = embeddings[index_positive2]
    negatives = torch.zeros((len(embeddings)-2, 128)).to(device)
    pos = 0
    for i, sample in enumerate(embeddings):
        if i != index_positive1 and i != index_positive2:
            negatives[pos] = sample
            pos += 1
    sim = torch.nn.CosineSimilarity(dim=0)
    numerator = torch.exp(sim(positive1, positive2)/temperature)
    denominator = 0
    for negative in negatives:
        denominator += torch.exp(sim(positive1, negative)/temperature)
    fraction = numerator/denominator
    loss = -torch.log(fraction)
    return loss


def batch_loss(outputs):
    loss = 0
    for index in range(batch_size):
        loss += NT_Xent_loss(outputs, 2*index, 2*index+1) + NT_Xent_loss(outputs, 2*index+1, 2*index)
    return loss/(2*batch_size)


def augment(sample):
    t1 = transforms.Compose({
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip()
    })
    t2 = transforms.Compose({
        transforms.ColorJitter(0.6, 0.6, 0.6, 0.3),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip()
    })
    augment1 = transforms.Resize((resize_x, resize_y))(random_crop(t1(sample), random_crop_size))
    augment2 = t2(sample)
    return augment1, augment2


def contrastive_learning(cl_loader, model):
    start = time.time()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate_cl)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    for epoch in range(num_epochs_cl):
        for i, (samples, _) in enumerate(cl_loader):
            samples = samples.to(device)
            augmentations = torch.zeros((2*batch_size, 3, resize_x, resize_y)).to(device)
            for index, (sample) in enumerate(samples):
                a1, a2 = augment(sample)
                augmentations[2*index] = a1
                augmentations[2*index+1] = a2
                
                fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
                a = sample.to("cpu")
                b = a1.to("cpu")
                c = a2.to("cpu")
                ax1.imshow(a.permute(1,2,0))
                ax2.imshow(b.permute(1,2,0))
                ax3.imshow(c.permute(1,2,0))
                plt.show()
            
            outputs = model(augmentations)
            
            loss = batch_loss(outputs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            print(f'Epoch {epoch+1}/{num_epochs_cl}\tStep: {i+1}/{len(cl_loader)}\tLoss:{loss:.4f}')
        scheduler.step()
    print(f'Time for learning: {float(time.time() - start) /60:.1f}m')
    return model


def finetune(finetune_loader, eval_loader, model):
    start = time.time()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate_ft)
    criterion = torch.nn.CrossEntropyLoss()
    num_epochs_ft = 0
    continue_finetune = True
    last_best_acc = 0
    counter_worse_acc = 0
    while continue_finetune:
        for step, (images, labels) in enumerate(finetune_loader):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            print(f'Epoch {num_epochs_ft+1}\tStep: {step+1}/{len(finetune_loader)}\tLoss:{loss:.4f}')
        num_epochs_ft += 1
        with torch.no_grad():
            n_correct_preds = 0
            n_samples = 0
            for images, labels in eval_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                n_samples += labels.shape[0]
                n_correct_preds += (preds == labels).sum().item()
            acc = n_correct_preds/n_samples
            if acc > last_best_acc:
                last_best_acc = acc
                counter_worse_acc = 0
                print(f'New best acc: {last_best_acc}')
                torch.save(model.state_dict(), FILEFT)
                print(f'Model saved in {FILEFT}')
            else:
                if counter_worse_acc == max_worse_acc_finetuner-1:
                    continue_finetune = False
                counter_worse_acc += 1
                print(f'Worse acc, {counter_worse_acc}th time in a row')
    print(f'Time for fine-tuning: {float(time.time() - start) /60:.1f}m')
    print(f'Epochs trained: {num_epochs_ft-10}')
    return model


def test(test_loader, model):
    n_correct_preds = 0
    n_samples = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        n_samples += labels.shape[0]
        n_correct_preds += (preds == labels).sum().item()
        
    print(f'Accuracy on test set: {100*n_correct_preds/n_samples:.1f}%')


if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Resize((resize_x, resize_y)),
        transforms.ToTensor()
    ])
    
    cl_dataset = datasets.ImageFolder('./images/cl/', transform=transform)
    finetune_dataset = datasets.ImageFolder('./images/finetune/', transform=transform)
    eval_dataset = datasets.ImageFolder('./images/eval', transform=transform)
    test_dataset = datasets.ImageFolder('./images/test/', transform=transform)
    
    cl_loader = data.DataLoader(cl_dataset, shuffle=True, batch_size=batch_size)
    finetune_loader = data.DataLoader(finetune_dataset, shuffle=True, batch_size=batch_size)
    eval_loader = data.DataLoader(eval_dataset, shuffle=False, batch_size=batch_size)
    test_loader = data.DataLoader(test_dataset, shuffle=False, batch_size=batch_size)
    
    dataset_sizes = {'cl': len(cl_dataset), 'finetune': len(finetune_dataset), 'eval': len(eval_dataset), 'test': len(test_dataset)}
    class_names = cl_dataset.classes
    print(f'Classes: {class_names}')
    print(f'Batch size: {batch_size}')
    print(f'Datasets sizes: {dataset_sizes}')
    
    
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    projection_head = ProjectionHead(2048, 2048, 128)
    model.fc = projection_head
    model = model.to(device)
    
    #######TRAINING CONTRASTIVE LEARNING###############
    model = contrastive_learning(cl_loader, model)
    torch.save(model.state_dict(), FILECL)
    print(f'Model saved in {FILECL}')
    ###################################################
    
    model.load_state_dict(torch.load(FILECL))
    print(f'Model loaded from {FILECL}')
    model.fc = FineTuneNN().to(device)
    
    #######FINETUNE MODEL##############################
    model = finetune(finetune_loader, eval_loader, model)
    ###################################################
    
    
    model.load_state_dict(torch.load(FILEFT))
    print(f'Model loaded from {FILEFT}')
    test(test_loader, model)