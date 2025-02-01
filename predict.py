from fastapi import FastAPI, File, UploadFile
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import io

# Initialize FastAPI app
app = FastAPI()

# Load the pretrained Inception V3 model with ImageNet weights
model = models.inception_v3(weights='IMAGENET1K_V1')

# Modify the final layer for binary classification
num_ftrs = model.fc.in_features  
model.fc = nn.Sequential(
    nn.Linear(num_ftrs, 1),  # Output 1 class for binary classification
    nn.Sigmoid()  # Activation for binary classification
)

# Move model to available device (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Load the saved model weights (trained with fc layer from train.py)
model_path = 'final_model.pt'  
model.load_state_dict(torch.load(model_path, map_location=device))  # Ensure correct device
model.eval()  # Set the model to evaluation mode for inference

# Define the preprocessing pipeline
preprocess = transforms.Compose([
    transforms.Resize((299, 299)), 
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize
])

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read the image file
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data)).convert("RGB")  # Open and convert to RGB

    # Preprocess the image
    input_tensor = preprocess(image).unsqueeze(0).to(device)  # Add batch dimension & move to device

    # Perform inference
    with torch.no_grad():
        output = model(input_tensor)  # Forward pass (sigmoid is already in model.fc)
        prediction = torch.round(output).item()  # Round to get binary prediction (0 or 1)

    return {"prediction": 'loaf' if int(prediction) == 1 else 'cat', "probability": float(output.item())}
