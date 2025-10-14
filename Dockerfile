# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /code

# Copy the requirements file into the container at /code
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
# Use --no-cache-dir to keep the image size smaller
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code into the container at /code
# This includes app.py, the python_services folder, and the model folder
COPY . .

# Make port 7860 available to the world outside this container
# This is the default port for Hugging Face Spaces
EXPOSE 7860

# Define the command to run your app.
# This will start the Uvicorn server, making your FastAPI app available.
# It's set to listen on all network interfaces (0.0.0.0).
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
