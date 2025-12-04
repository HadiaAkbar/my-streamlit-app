import pandas as pd
import kagglehub
import os

# Download dataset from Kaggle
print("Downloading dataset...")
path = kagglehub.dataset_download("clmentbisaillon/fake-and-real-news-dataset")

print("Dataset downloaded to:", path)

# List files
for root, dirs, files in os.walk(path):
    for file in files:
        print(os.path.join(root, file))