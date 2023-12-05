import torch
import torch.nn.functional as F

# 4x 5 x3
tensor = torch.tensor([
    # Batch 1: Distinct directions
    [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [0, 1, 1]],
    # Batch 2: Varying magnitudes and directions
    [[1, 2, 3], [4, 5, 6], [10, 0, 0], [0, 10, 0], [0, 0, 10]],
    # Batch 3: Negative and positive values
    [[-1, -1, 0], [1, 1, 0], [1, -1, 0], [0, 1, -1], [0, -1, 1]],
    # Batch 4: Mixed values
    [[1, 2, -2], [-1, -2, 2], [3, -3, 1], [-3, 3, -1], [2, -2, 3]]
], dtype=torch.float32)

# Function to calculate pairwise cosine similarities
def pairwise_cosine_similarities(batch):
    # Normalize the batch to unit vectors along the last dimension
    normalized_batch = F.normalize(batch, p=2, dim=1)
    # Compute cosine similarity for each pair
    num_candidates = normalized_batch.shape[0]
    similarity_matrix = torch.zeros((num_candidates, num_candidates))
    for i in range(num_candidates):
        for j in range(num_candidates):
            similarity_matrix[i, j] = F.cosine_similarity(normalized_batch[i], normalized_batch[j], dim=0)
    return similarity_matrix

# Function to find the tensor of pairs with the minimum cosine similarity (maximum distance)
def min_cosine_similarity_pairs(tensor):
    max_pairs_tensor = torch.empty(tensor.size(0), 2, tensor.size(2))

    for i, batch in enumerate(tensor):
        similarities = pairwise_cosine_similarities(batch)
        # Fill diagonal with a high value to ignore self-similarities
        similarities.fill_diagonal_(2)
        # Find the indices of the minimum cosine similarity
        min_idx = torch.argmin(similarities)
        min_pair_indices = (min_idx // similarities.size(1), min_idx % similarities.size(1))
        max_pairs_tensor[i] = torch.stack((batch[min_pair_indices[0]], batch[min_pair_indices[1]]))

    return max_pairs_tensor

# Apply the function to the entire tensor
min_pairs_tensor = min_cosine_similarity_pairs(tensor)

print('max_pairs_tensor', min_pairs_tensor)



z1, z2 = min_pairs_tensor[:,0], min_pairs_tensor[:,1]

print(min_pairs_tensor.shape)
print(z1)
print(z2)