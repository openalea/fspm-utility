import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from openalea.mtg import MTG

# Ensure you have OpenAlea.MTG installed: pip install openalea.mtg
def extract_mtg_data(mtg, time_step):
    """
    Extracts node features and edges from the MTG at a given time step.
    :param mtg: The MTG graph object from OpenAlea.
    :param time_step: Current time step for which to extract data.
    :return: Node features tensor and edge index tensor.
    """
    node_features = []
    edges = []
    
    # Iterate over nodes in the MTG at the current time step (scale = 1 for current growth level)
    for vid in mtg.vertices_iter(scale=1):  
        node_data = mtg.property('time_series_data')[vid][time_step]  # Extract features for the current time step
        node_features.append(node_data)
        
        # Extract edges (parent-child relationships in the tree structure)
        for child in mtg.children(vid):
            edges.append((vid, child))
    
    # Convert lists to PyTorch tensors
    node_features = torch.tensor(node_features, dtype=torch.float)
    edges = torch.tensor(edges, dtype=torch.long)
    
    return node_features, edges


class UnifiedAttentionDynamic(nn.Module):
    def __init__(self, in_channels):
        """
        Unified attention mechanism that dynamically handles varying node counts.
        :param in_channels: Number of input features per node.
        """
        super(UnifiedAttentionDynamic, self).__init__()
        self.in_channels = in_channels  # Feature dimension
        
    def forward(self, x):
        """
        Forward pass for unified attention.
        :param x: Input tensor of shape [num_nodes, seq_len, in_channels].
        :return: Attended input and attention scores.
        """
        num_nodes, seq_len, in_channels = x.shape
        
        # Initialize learnable attention weights for current node size
        attn_weights = nn.Parameter(torch.randn(num_nodes, seq_len, in_channels))
        
        # Apply softmax over the feature dimension (last dimension)
        attn_scores = F.softmax(attn_weights, dim=-1)  # Normalize over the feature dimension
        
        # Apply attention scores to the input data
        attended_x = attn_scores * x
        
        return attended_x, attn_scores


class A3TGCNUnifiedAutoencoderDynamic(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        """
        A3TGCN-based autoencoder with unified attention for dynamic graphs.
        :param in_channels: Number of input features per node.
        :param hidden_channels: Latent dimensionality (hidden size).
        :param out_channels: Output dimensionality (should match input features for reconstruction).
        """
        super(A3TGCNUnifiedAutoencoderDynamic, self).__init__()
        self.a3tgcn = A3TGCN(in_channels, hidden_channels, seq_len=None, num_nodes=None)
        self.unified_attention = UnifiedAttentionDynamic(in_channels)
        self.decoder = nn.Linear(hidden_channels, out_channels)
    
    def forward(self, x_seq, edge_index_seq):
        """
        Forward pass of the model.
        :param x_seq: Input node features tensor.
        :param edge_index_seq: Edge index tensor for the current time step's graph structure.
        :return: Reconstructed input and attention scores.
        """
        # Apply unified attention to the input
        attended_x_seq, attn_scores = self.unified_attention(x_seq)
        
        # Pass through A3TGCN to get the latent representation
       
if __name__ == "__main__":
    # Define model and optimizer
    in_channels = 5  # Example feature dimension, should match your actual input feature size
    hidden_channels = 64  # Latent space size
    out_channels = in_channels  # Output size should match input size for reconstruction

    model = A3TGCNUnifiedAutoencoderDynamic(in_channels, hidden_channels, out_channels)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Example MTG object loaded from file
    # mtg = MTG('path_to_mtg_file.mtg')  # Load your actual MTG file

    # Training loop
    num_epochs = 50
    max_time_steps = 10  # Example time steps, adjust based on your data
    model.train()

    for epoch in range(num_epochs):
        total_loss = 0
        
        for time_step in range(max_time_steps):  # Iterate over all time steps
            # Dynamically extract node features and edges for the current time step
            node_features, edges = extract_mtg_data(mtg, time_step)
            
            # Zero the gradients
            optimizer.zero_grad()
            
            # Forward pass through the model
            reconstructed_x_seq, _ = model(node_features, edges)
            
            # Compute reconstruction loss (MSE between input and reconstructed features)
            loss = F.mse_loss(reconstructed_x_seq, node_features)
            loss.backward()  # Backpropagate the loss
            optimizer.step()  # Update the model parameters
            
            total_loss += loss.item()
        
        avg_loss = total_loss / max_time_steps
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.6f}')

    #Visualizing the attention weights

    model.eval()

    # Example visualization of attention weights for each time step
    for time_step in range(max_time_steps):
        node_features, edges = extract_mtg_data(mtg, time_step)
        
        # Forward pass to get the attention scores
        _, attn_scores = model(node_features, edges)
        
        # Plot attention scores averaged over features
        plt.figure()
        plt.imshow(attn_scores.mean(dim=-1).cpu().detach().numpy(), cmap='viridis', aspect='auto')
        plt.title(f'Attention Weights at Time Step {time_step}')
        plt.xlabel('Node Index')
        plt.ylabel('Time Step')
        plt.colorbar(label='Attention Weight')
        plt.show()
