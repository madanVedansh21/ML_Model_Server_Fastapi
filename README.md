# Transformer Health Monitoring API (FastAPI)

This repository hosts a unified FastAPI server that serves two different AI models for monitoring the health of electrical transformers:

1.  **SCADA Model**: Predicts transformer faults based on real-time SCADA (Supervisory Control and Data Acquisition) sensor data.
2.  **FRA Model**: Analyzes Frequency Response Analysis (FRA) data to detect mechanical and electrical faults.

The project is structured to run as a single application, making it easy to test locally and deploy on platforms like Hugging Face Spaces.

## Project Structure

- `app.py`: The main FastAPI application that serves as the single entry point.
- `python_services/`: Contains the core logic for the SCADA and FRA prediction pipelines.
- `model/`: Contains the pre-trained model files, scalers, and encoders for both services.
- `samples/`: Contains sample data that can be used to test the API endpoints.
- `requirements.txt`: A single file listing all Python dependencies for the project.

## Available Endpoints

The server exposes the following endpoints:

### SCADA

- `POST /scada/predict-json`: Accepts a JSON object with SCADA sensor readings and returns a detailed diagnosis, including fault type, severity, and recommended actions.

### FRA

- `POST /fra/predict-file`: Accepts a CSV file with FRA measurement data and returns a diagnosis.
- `POST /fra/predict-json`: Accepts a JSON payload containing FRA data (either as arrays or as inline CSV content) and returns a diagnosis.

## How to Run the Server Locally

Follow these steps to set up and run the application on your local machine.

### 1. Prerequisites

- Python 3.9+

### 2. Setup (Windows PowerShell)

First, clone the repository to your local machine.

Then, from the root directory of the project (`ML_Model_Server_Fastapi`), run the following commands:

```powershell
# Create a new virtual environment
python -m venv .venv

# Activate the virtual environment
.\.venv\Scripts\Activate.ps1

# Install all required dependencies
pip install -r requirements.txt
```

### 3. Run the Server

Once the setup is complete, run the main application using Uvicorn:

```powershell
# Make sure you are in the root directory of the project
uvicorn app:app --host 0.0.0.0 --port 8000
```

The server will now be running and accessible at `http://localhost:8000`.

## How to Test the API

You can test the running server by sending requests to its endpoints from a new terminal.

### Test the SCADA Endpoint

Use the sample SCADA data to get a prediction. This command sends the content of `scada-data.json` to the prediction endpoint.

```powershell
Get-Content 'samples\scada-data.json' -Raw | Invoke-RestMethod -Uri http://localhost:8000/scada/predict-json -Method Post -ContentType 'application/json'
```

### Test the FRA Endpoint

Use the sample FRA data file to get a prediction.

```powershell
Invoke-RestMethod -Uri http://localhost:8000/fra/predict-file -Method Post -InFile 'samples\fra_validated_dataset_20251014_171555.csv'
```

You should receive a JSON response in your terminal with the model's diagnosis and recommendations.
