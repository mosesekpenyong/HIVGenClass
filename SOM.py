import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.preprocessing import MinMaxScaler
from minisom import MiniSom

# Load data
data = np.loadtxt('features_new.csv', delimiter=',')  # Assuming 'features_new.csv' contains the data

# Normalize data
scaler = MinMaxScaler()
data_normalized = scaler.fit_transform(data)

# Create a Self-Organizing Map
lattice_size = [20, 20]
cover_steps = 100
init_neighbor = 3
topology = 'hexagonal'
distance = 'euclidean'
som = MiniSom(lattice_size[0], lattice_size[1], data.shape[1], sigma=init_neighbor, learning_rate=0.5,
              topology=topology, random_seed=42)

# Initialize weights
som.random_weights_init(data_normalized)

# Train the SOM
som.train_random(data_normalized, cover_steps)

# Get the clustered labels
cluster_labels = som.win_map(data_normalized)

# Plot clusters
plt.figure(figsize=(10, 10))
for position, values in cluster_labels.items():
    plt.scatter(position[0], position[1], color=plt.cm.plasma(len(values) / len(data)), marker='o')
plt.title('Self-Organizing Map Clusters')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()