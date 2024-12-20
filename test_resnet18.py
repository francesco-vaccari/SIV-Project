import torch
import torchvision
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

device = torch.device("mps")

model = torchvision.models.resnet18()
model.fc = torch.nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load("saves/resnet18.pth"))
model.to(device)

data = torchvision.datasets.ImageFolder("./images/finetune", transform=torchvision.transforms.Compose([
        torchvision.transforms.Resize((255, 255)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]))
loader = DataLoader(data, batch_size=4, shuffle=True)
print(f'Images found: {len(loader)*4}')

total = 0
correct = 0
for image, label in loader:
    image = image.to(device)
    label = label.to(device)
    output = model(image)
    for i, elem in enumerate(output):
        _, pred = torch.max(elem, 0)
        # image = image[0].permute(1,2,0)
        # plt.imshow(image)
        if(label[i] == pred):
            correct += 1
        # else:
        #     image = image[0].to(torch.device("cpu")).permute(1,2,0)
        #     plt.imshow(image)
        #     plt.show()
        total += 1
        # plt.show()

print(f'Acc: {float(correct/total)*100:.1f}%')
