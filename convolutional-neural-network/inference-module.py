# import libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import numpy as np

# Define the neural network class
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)  # Adjust the input size based on your image dimensions
        self.fc2 = nn.Linear(128, 9)

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)  # Adjust the size based on your image dimensions
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# ask user for test input
print("Enter the path of the image you want to test:")
test_image_path = input()

# Define data transformations to apply to the images when loaded
transform = transforms.Compose([
    transforms.Resize((32, 32)),  # Resize images to a common size
    transforms.ToTensor(),  # Convert images to PyTorch tensors
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # Normalize pixel values
])

# create image dataset
test_image_dataset = ImageFolder(root=test_image_path, transform=transform)

# get labels from file names
test_image_labels = []
for file_name in test_image_dataset.samples:
    test_image_labels.append(file_name[0].split('/')[-1].split('\\')[3].split('_')[0])
test_image_labels = np.array(test_image_labels)

unique_labels, label_indices = np.unique(test_image_labels, return_inverse=True) # Convert labels to unique indices
label_mapping = dict(zip(range(len(unique_labels)), unique_labels)) # Create a mapping from label indices to labels
test_image_dataset.targets = torch.tensor(label_indices) # Add label indices to the dataset
test_image_dataset.classes = unique_labels # Change training classes to unique labels
test_image_dataset.class_to_idx = label_mapping # Change class_to_idx dictionary to match new labels
test_image_dataset.idx_to_class = {v: k for k, v in label_mapping.items()} # Change idx_to_class dictionary to match new labels
test_image_dataset.samples = [list(elem) for elem in test_image_dataset.samples]
for i in range(len(test_image_dataset.samples)):
    test_image_dataset.samples[i][1] = label_indices[i]

# reassigned test_image_loader
test_image_loader = DataLoader(test_image_dataset, batch_size=250, shuffle=True)

# load the model from disk
model = Net()
model.load_state_dict(torch.load('0602-668913771-Hashmi.pth'))

# Define the device for training
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# Move the model to the device specified above
model.to(device)

# test the model on the test image
correct_test_image = 0
total_test_image = 0

with torch.no_grad():
    model.eval()
    for data in test_image_loader:
        inputs, targets = data
        inputs = inputs.to(device)
        targets = targets.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total_test_image += targets.size(0)
        correct_test_image += (predicted == targets).sum().item()

test_image_accuracy = correct_test_image / total_test_image
print(f'Test Image Accuracy: {100 * test_image_accuracy:.2f}%')

exit()