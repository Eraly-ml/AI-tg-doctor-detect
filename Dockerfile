FROM python:3.10-slim

# Install system dependencies for SciPy
RUN apt-get update && apt-get install -y libblas-dev liblapack-dev gfortran

# Upgrade pip and setuptools
RUN pip install --upgrade pip setuptools

# Install required Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install SciPy separately (in case of issues with other dependencies)
RUN pip install --no-cache-dir scipy

# Copy the rest of the application code
COPY . /app

WORKDIR /app
CMD ["python", "bot.py"]
