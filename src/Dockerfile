# This is a sample Image 
FROM pytorch/pytorch:2.2.0-cuda11.8-cudnn8-devel

# Install the requirments
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

# Copy the code
ADD data_management/ data_management/
ADD helpers/ helpers/
ADD modules/ modules/
ADD Model.py Model.py
ADD GPUtraining/ GPUtraining/

# Run segmentation scripts.py
# CMD python GPUtraining/trainGAT.py
