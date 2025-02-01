# Cat or Loaf Image Classification Project

<p align="center">
  <img src="custom_test_img/loaf.png" alt="Display Loaf">
</p>

This project uses the pre-trained **Inception V3** CNN model to classify images of "catloafs" (cats lying with their paws tucked under them) from default cats. It is implemented using FastAPI for serving predictions and utilizes Docker for easy deployment. The project also includes scripts for training the model and testing it using a simple Python script.

## Table of Contents
1. [Description](#description)
2. [How to Clone the Repository](#how-to-clone-the-repository)
3. [Setting Up the Environment](#setting-up-the-environment)
4. [Training the Model with `train.py`](#training-the-model-with-trainpy)
5. [Building and Running the Docker Container](#building-and-running-the-docker-container)
6. [How to Use `test.py`](#how-to-use-testpy)
7. [Minikube Setup and Kubernetes Deployment](#minikube-setup-and-kubernetes-deployment)

## Description

The Catloaf Image Classification project aims to classify images into "catloaf" or "non-catloaf" categories. The project uses a Convolutional Neural Network (CNN) based model, trained using the Inception V3 architecture, which is then deployed using FastAPI and Docker.

The key features of the project are:
- **FastAPI** for serving predictions.
- **Docker** for containerization, making it easy to run anywhere.
- **PyTorch** for training the model using a custom dataset of catloaf images.

## How to Clone the Repository

To get started, clone the GitHub repository to your local machine using the following command:

```bash
git clone https://github.com/kabsmeiou/catloaf.git
```

Navigate to the project folder:

```bash
cd catloaf
```

## Setting Up the Environment

To set up the environment in Conda, you can use the provided `environment.yml` file. This file contains all the dependencies required to run the project.

1. First, create a new Conda environment with the dependencies specified in the `environment.yml`:

```bash
conda env create -f environment.yml
```

2. Activate the Conda environment:

```bash
conda activate pytorch-env
```

After activating the environment, you can proceed with building and running the Docker container, or you can use the provided scripts directly in the Conda environment.


## Training the Model with `train.py`

Before running the Docker image, you can train the model using the `train.py` script. Follow these steps to train the model:

1. Ensure you have the necessary dependencies installed (you can use the `environment.yml` to set up your environment).
2. Run the `train.py` script to train the model:

```bash
python train.py
```

This will train the model on your dataset and save the trained model to a file (e.g., `final_model.pt`).

Once the model is trained, you can run the Docker image as described earlier to serve the trained model via FastAPI.

## Building and Running the Docker Container

To build and run the Docker container, follow these steps:

1. Build the Docker image:

```bash
docker build -t catloaf-image .
```

2. Run the Docker container:

```bash
docker run -p 8000:8000 catloaf-image
```

This will start the FastAPI server, which you can access at `http://localhost:8000`.

## How to Use `test.py`

Once the Docker container is running, you can use the `test.py` script to send images to the FastAPI server for prediction.

1. Make sure the Docker container is running.
2. Run the `test.py` script with the following command:

```bash
python test.py
```

The script will send an image to the server and print the classification result (either "cat" or "loaf") along with the probability of the classification.


## Minikube Setup and Kubernetes Deployment

<video width="800" height="450" controls>
  <source src="kubernetes_vid.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

### 1. Start Minikube

Start a local Kubernetes cluster with Docker as the driver:

```bash
minikube start --driver=docker
```

Verify the cluster:

```bash
minikube status
kubectl get nodes
```

### 2. Build and Load the Docker Image

Your image `catloaf-image` must be available to Minikube's Docker daemon.

**Step 1:** Point your shell to Minikube’s Docker daemon

```bash
eval $(minikube docker-env)
```

**Step 2:** Build the Docker image (from your project directory)

```bash
docker build -t catloaf-image .
```

**Step 3:** Verify the image is in Minikube’s Docker registry

```bash
docker images | grep catloaf-image
```

### 3. Deploy to Kubernetes

Apply the manifest file:

```bash
kubectl apply -f deployment.yaml
```

Verify the deployment and service:

```bash
kubectl get deployments
kubectl get pods
kubectl get services
```

### 4. Access the Application

**Option 1:** Use minikube service to get the URL

```bash
minikube service catloaf-service --url
```

### 5. Clean Up

Delete the deployment and service:

```bash
kubectl delete -f deployment.yaml
```

Stop Minikube:

```bash
minikube stop
```

Delete Minikube cluster (optional):

```bash
minikube delete
```

