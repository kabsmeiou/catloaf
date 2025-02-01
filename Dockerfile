# Use Miniconda as the base image
FROM continuumio/miniconda3:latest

# Set the working directory
WORKDIR /app

# Copy the Conda environment file
COPY environment.yml .

# Create the Conda environment
RUN conda env create -f environment.yml

# Ensure the environment is activated correctly
SHELL ["conda", "run", "-n", "pytorch-env", "/bin/bash", "-c"]

# Copy application code
COPY . .

# Expose the port for FastAPI
EXPOSE 8000

# Command to run the FastAPI application
CMD ["conda", "run", "-n", "pytorch-env", "uvicorn", "predict:app", "--host", "0.0.0.0", "--port", "8000"]
