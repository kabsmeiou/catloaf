import torch
import torch.nn as nn
from torchvision import models
from torchvision.transforms import v2
from PIL import Image

# Load the pretrained Inception V3 model with ImageNet weights
model = models.inception_v3(weights='IMAGENET1K_V1')

# Modify the final layer for binary classification
num_ftrs = model.fc.in_features  
model.fc = nn.Sequential(
    nn.Linear(num_ftrs, 1),  # Output 1 class for binary classification
    nn.Sigmoid()  # Activation for binary classification
)

# Load the saved model weights (trained with fc layer from train.py)
model_path = 'final_model.pt'  
model.load_state_dict(torch.load(model_path))  # Load the trained weights
model.eval()  # Set the model to evaluation mode for inference

# Move model to GPU if available to increase performance
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Define preprocessing pipeline (must match the training pipeline)
preprocess = v2.Compose([
    v2.Resize(size=(299, 299)),  # Resize image to 299x299 as InceptionV3 expects
    v2.ToImage(),  
    v2.ToDtype(torch.float32, scale=True), # Convert image to tensor
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load and preprocess an sample image
image_path = 'custom_test_img/cat_loaf_script.jpg'
image = Image.open(image_path).convert('RGB') 
input_tensor = preprocess(image).unsqueeze(0)  

input_tensor = input_tensor.to(device)

# Perform inference
with torch.no_grad():  # Disable gradient calculation
    output = model(input_tensor) 
    prediction = output.item()  

# Print the result
print(f'Expected Result: Loaf')
print(f'Prediction: {prediction:.4f} (Loaf)' if prediction > 0.5 else f'Prediction: {prediction:.4f} (Cat)')

print('*' * 20)

# Load and preprocess an sample image
image_path = 'custom_test_img/cat_not_loaf.jpg'  
image = Image.open(image_path).convert('RGB')
input_tensor = preprocess(image).unsqueeze(0)  

input_tensor = input_tensor.to(device)

# Perform inference
with torch.no_grad(): 
    output = model(input_tensor)  
    prediction = output.item()  


print(f'Expected Result: Cat')
print(f'Prediction: {prediction:.4f} (Loaf)' if prediction > 0.5 else f'Prediction: {prediction:.4f} (Cat)')
