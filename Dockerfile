FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    curl \
    sudo \
    dpkg \
    && rm -rf /var/lib/apt/lists/*

# Set workdir
WORKDIR /app

# Copy the codebase
COPY . /app

# Install Python dependencies
RUN pip install --upgrade pip && pip install -r requirements.txt

# Install MMT tools using deb packages
RUN sudo dpkg -i mmt-packages/mmt-dpi_1.7.4_c5a4a6b_Linux_x86_64.deb || true
RUN sudo dpkg -i mmt-packages/mmt-security_1.2.14_d74aea4_Linux_x86_64.deb || true
RUN sudo dpkg -i mmt-packages/mmt-probe_1.5.5_6765397_Linux_x86_64_pcap.deb || true

# Expose FastAPI port
EXPOSE 8000

# Start the FastAPI server
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
