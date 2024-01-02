############## ASSIGNMENT 5 ##############
## ZOHAIR HASHMI | 668913771 | zhashm4@uic.edu

### Importing Libraries
import os
import shutil
from tqdm import tqdm
import zipfile
import numpy as np

import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim


### **ZIP FILE EXTRACTION**
zip_file_path = "geometry_dataset.zip"
dataset_path = "./geometry_dataset"

# Get the total number of files in the zip archive
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    file_list = zip_ref.namelist()
    total_files = len(file_list)

# Use tqdm to track progress during extraction
with tqdm(total=total_files, desc="Extracting files", unit="file") as pbar:
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        for file in file_list:
            zip_ref.extract(file, dataset_path)
            pbar.update(1)


### **TRAIN TEST SPLIT**
# Sort the files in the dataset folder
files = os.listdir(os.path.join(dataset_path, 'output'))
files.sort()

# Create two folders for train and test
train_path = os.path.join(dataset_path, "train")
test_path = os.path.join(dataset_path, "test")
os.makedirs(train_path, exist_ok=True)
os.makedirs(test_path, exist_ok=True)

# Copy 8000 .png images of each class to train & 2000 .png images to test folder
for i in range(9):

    # Train Folder
    for j in tqdm(range(8000), desc=f'Copying files to train (Class {i+1})'):
        source_path = os.path.join(dataset_path, "output", files[i*10000 + j])
        destination_path = os.path.join(train_path, "output", files[i*10000 + j])
        shutil.copy(source_path, destination_path)

    # Test Folder
    for j in tqdm(range(2000), desc=f'Copying files to test (Class {i+1})'):
        source_path = os.path.join(dataset_path, "output", files[i*10000 + j + 8000])
        destination_path = os.path.join(test_path, "output", files[i*10000 + j + 8000])
        shutil.copy(source_path, destination_path)

### **DATA PROCESSING**
# Define data transformations to apply to the images when loaded
transform = transforms.Compose([
    transforms.Resize((32, 32)),  # Resize images to a common size
    transforms.ToTensor(),  # Convert images to PyTorch tensors
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # Normalize pixel values
])

# Define the paths to your training and test data folders
train_path = os.path.join(dataset_path, "train")
test_path = os.path.join(dataset_path, "test")

train_dataset = ImageFolder(root=train_path, transform=transform)
test_dataset = ImageFolder(root=test_path, transform=transform)

# Get labels from file names and assign then to the respective datasets
train_labels = []
for file_name in train_dataset.samples:
    train_labels.append(file_name[0].split('/')[-1].split('\\')[3].split('_')[0])
train_labels = np.array(train_labels)

unique_labels, label_indices = np.unique(train_labels, return_inverse=True) # Convert labels to unique indices
label_mapping = dict(zip(range(len(unique_labels)), unique_labels)) # Create a mapping from label indices to labels
train_dataset.targets = torch.tensor(label_indices) # Add label indices to the dataset
train_dataset.classes = unique_labels # Change training classes to unique labels
train_dataset.class_to_idx = label_mapping # Change class_to_idx dictionary to match new labels
train_dataset.idx_to_class = {v: k for k, v in label_mapping.items()} # Change idx_to_class dictionary to match new labels
train_dataset.samples = [list(elem) for elem in train_dataset.samples]
for i in range(len(train_dataset.samples)):
    train_dataset.samples[i][1] = label_indices[i]

test_labels = []
for file_name in test_dataset.samples:
    test_labels.append(file_name[0].split('/')[-1].split('\\')[3].split('_')[0])
test_labels = np.array(test_labels)

unique_labels, label_indices = np.unique(test_labels, return_inverse=True) # Convert labels to unique indices
label_mapping = dict(zip(range(len(unique_labels)), unique_labels)) # Create a mapping from label indices to labels
test_dataset.targets = torch.tensor(label_indices) # Add label indices to the dataset
test_dataset.classes = unique_labels # Change training classes to unique labels
test_dataset.class_to_idx = label_mapping # Change class_to_idx dictionary to match new labels
test_dataset.idx_to_class = {v: k for k, v in label_mapping.items()} # Change idx_to_class dictionary to match new labels
test_dataset.samples = [list(elem) for elem in test_dataset.samples]
for i in range(len(test_dataset.samples)):
    test_dataset.samples[i][1] = label_indices[i]

# Create Train and Test data loaders
train_loader = DataLoader(train_dataset, batch_size=250, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=250, shuffle=True)


## **MODEL DEFINITION**
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

model = Net()
model = nn.DataParallel(model) # Wrap model for Parallel processing

loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

model.to(device) # moving model to cuda

train_losses = []
test_losses = []
train_accuracies = []
test_accuracies = []

for epoch in range(20):
    print(f'Starting epoch {epoch+1}')

    current_loss = 0.0 # Reset current loss to zero for each epoch

    # Model Training
    model.train()
    for i, data in enumerate(tqdm(train_loader)): # batch wise training
        inputs, targets = data
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs, targets)
        loss.backward()
        optimizer.step()

        current_loss += loss.item()
        if i % 500 == 499:
            print('Loss after mini-batch %5d: %.3f' % (i + 1, current_loss / 500))
            current_loss = 0.0

    # Calculating train accuracy
    correct_train = 0
    total_train = 0
    with torch.no_grad():
        model.eval()
        for data in train_loader:
            inputs, targets = data
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total_train += targets.size(0)
            correct_train += (predicted == targets).sum().item()

    train_accuracy = correct_train / total_train
    print(f'Training Accuracy after epoch {epoch+1}: {100 * train_accuracy:.2f}%')

    # Calculating test accuracy
    correct_test = 0
    total_test = 0
    with torch.no_grad():
        model.eval()
        for data in test_loader:
            inputs, targets = data
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total_test += targets.size(0)
            correct_test += (predicted == targets).sum().item()

    test_accuracy = correct_test / total_test
    print(f'Test Accuracy after epoch {epoch+1}: {100 * test_accuracy:.2f}%')

    train_losses.append(current_loss / 500)
    test_losses.append(loss_function(outputs, targets).item())
    train_accuracies.append(train_accuracy)
    test_accuracies.append(test_accuracy)

# Save the model to disk
torch.save(model.state_dict(), '0602-668913771-Hashmi.pth.pth')

# Store the loss & accuracy to txt file
with open('train_losses.txt', 'w') as f:
    for item in train_losses:
        f.write("%s\n" % item)

with open('test_losses.txt', 'w') as f:
    for item in test_losses:
        f.write("%s\n" % item)

with open('train_accuracies.txt', 'w') as f:
    for item in train_accuracies:
        f.write("%s\n" % item)

with open('test_accuracies.txt', 'w') as f:
    for item in test_accuracies:
        f.write("%s\n" % item)


exit()