import torch

# 4x 5 x3
tensor = torch.tensor([
    [[1, 2, 3], [2, 3, 4], [3, 4, 5], [6, 7, 8], [9, 10, 11]],  # Batch 1
    [[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4], [5, 5, 5]],    # Batch 2
    [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]],    # Batch 3
    [[9, 8, 7], [8, 7, 6], [7, 6, 5], [6, 5, 4], [5, 4, 3]]     # Batch 4
], dtype=torch.float32)

# Function to calculate pairwise distances
def pairwise_distances(batch):
    # Compute pairwise distance matrix
    distance_matrix = torch.cdist(batch, batch, p=2)
    return distance_matrix

# Function to find the tensor of pairs with the maximum distance
def max_distance_pairs(tensor):
    max_pairs_tensor = torch.empty(tensor.size(0), 2, tensor.size(2))

    for i, batch in enumerate(tensor):
        distances = pairwise_distances(batch)
        # Fill diagonal with -inf to ignore self-distances
        distances.fill_diagonal_(-float('inf'))
        # Find the indices of the maximum distance
        max_idx = torch.argmax(distances)
        max_pair_indices = (max_idx // distances.size(1), max_idx % distances.size(1))
        max_pairs_tensor[i] = torch.stack((batch[max_pair_indices[0]], batch[max_pair_indices[1]]))

    return max_pairs_tensor

# Apply the function to the entire tensor
max_pairs_tensor = max_distance_pairs(tensor)

print('max_pairs_tensor',max_pairs_tensor)



z1, z2 = max_pairs_tensor[:,0], max_pairs_tensor[:,1]

print(max_pairs_tensor.shape)
print(z1)
print(z2)