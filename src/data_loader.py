import os
import numpy as np

def load_ransomware_data(data_path):
    data = []
    labels = []
    
    for filename in os.listdir(data_path):
        file_path = os.path.join(data_path, filename)
        with open(file_path, 'rb') as f:
            content = f.read()
            data.append(np.frombuffer(content, dtype=np.uint8))
            labels.append(0)  # Default label, replace with your logic
            
    data = np.array(data)
    labels = np.array(labels)
    return data, labels
