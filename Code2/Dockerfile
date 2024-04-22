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

# Create folders to store model and training results
RUN mkdir /models
RUN mkdir /results

# Copy and rename exp_get_explanation_step_1.py to main.py
COPY GPUtraining/trainGAT.py main.py

# Run main.py
CMD python main.py