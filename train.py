import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchvision.transforms import ToTensor, v2
import os
import random
from pathlib import Path

# Paths
train_dir = 'dataset/train'
test_dir = 'dataset/test'


# Credit: https://github.com/sberbank-ai/ru-dalle/blob/e96631a867fcadcfaa52eecb20b1e42b88aa4386/rudalle/utils.py
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

seed_everything(42)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(0)

import time
from tempfile import TemporaryDirectory

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    # Create a temporary directory to save training checkpoints
    with TemporaryDirectory() as tempdir:
        best_model_params_path = os.path.join(tempdir, 'best_model_params.pt')

        torch.save(model.state_dict(), best_model_params_path)
        best_acc = 0.0

        for epoch in range(num_epochs):
            print(f'Epoch {epoch + 1}/{num_epochs}')
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()   # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        
                        if isinstance(model, models.inception.Inception3) and phase == 'train':
                            outputs = outputs.logits
                            
                        #print(outputs)
                        preds = (outputs > 0.5).long() # For binary classification, use a threshold of 0.5
                        loss = criterion(outputs, labels.unsqueeze(1).float())
                        
                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.unsqueeze(1))

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]
                # print("Predictions: ", preds, '\n', "Labels: ", labels.unsqueeze(1), '\n', "\nTest: ", running_corrects.double())
                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
                
                if phase == 'val':
                    scheduler.step(epoch_loss)
                    
                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    torch.save(model.state_dict(), best_model_params_path)

            print()

        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val Acc: {best_acc:4f}')

        # load best model weights
        model.load_state_dict(torch.load(best_model_params_path, weights_only=True))
    return model

# #### Inception V3

# common classificaton pipeline from
# https://pytorch.org/vision/stable/auto_examples/transforms/plot_transforms_getting_started.html#sphx-glr-auto-examples-transforms-plot-transforms-getting-started-py
train_transforms = v2.Compose([
    # v2.RandomResizedCrop(size=(224, 224), antialias=True),
    # v2.RandomHorizontalFlip(p=0.5),
    v2.Resize(size=(299, 299)), # resizing directly because data augmentation has already been done on the dataset
    v2.ToImage(), # To image and dtype torch.float32 becuase v2.ToTensor() is deprecated
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

val_transforms = v2.Compose([
    # v2.RandomResizedCrop(size=(224, 224), antialias=True),
    # v2.RandomHorizontalFlip(p=0.5),
    v2.Resize(size=(299, 299)),
    v2.CenterCrop(size=(299, 299)),
    v2.ToImage(), # To image and dtype torch.float32 becuase v2.ToTensor() is deprecated
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load datasets
train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transforms)
test_dataset = datasets.ImageFolder(root=test_dir, transform=val_transforms)

# DataLoader (for batching and shuffling)
train_loader = DataLoader(
    train_dataset, 
    batch_size=16, 
    shuffle=True, 
    num_workers=4,
    worker_init_fn=seed_worker,
    generator=g
)
# Ensure workers have deterministic behavior
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

dataloaders = {
    'train': train_loader,
    'val': test_loader
}

dataset_sizes = {
    'train': len(train_loader.dataset),
    'val':len(test_loader.dataset)
}

model = models.inception_v3(weights='IMAGENET1K_V1')
num_ftrs = model.fc.in_features  # The input features to the final fully connected layer

# Replace the fully connected layer
# by default, inception_v3 has dropout of 0.5
model.fc = nn.Sequential(
    nn.Linear(num_ftrs, 1),  # Output 1 class for binary classification
    nn.Sigmoid()  # Activation for binary classification
)

# move to cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# loss and optimizer
criterion = nn.BCELoss()  # Binary Cross-Entropy Loss
optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.001)

# reduce lr on plateau
# https://medium.com/data-scientists-diary/guide-to-pytorch-learning-rate-scheduling-b5d2a42f56d4
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 
    mode='min', 
    factor=0.5, patience=5, 
    cooldown=2, 
    threshold=0.01
) # it seems that for smaller datasets, it is better to use a lr_scheduler

# freeze layers and unfreeze last layer
for param in model.parameters():
    param.requires_grad = False

for param in model.fc.parameters():
    param.requires_grad = True

model = train_model(model, criterion, optimizer, scheduler, num_epochs=10)

final_model = model

print(f"Finished training model: {final_model}")

# save final model
torch.save(final_model.state_dict(), 'final_model.pt')

# # ### Cat Test
# visualize_model_predictions(
#     final_model,
#     img_path='custom_test_img/cat.jpeg'
# )

# # ### Loaf Test
# visualize_model_predictions(
#     final_model,
#     img_path='custom_test_img/loaf.png'
# )

