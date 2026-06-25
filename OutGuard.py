import torch
import torch.nn as nn
import torch.nn.functional as F


class OutGuard(nn.Module):

    def __init__(self, input_dim=4096, projection_dim=128):
        super(OutGuard, self).__init__()
        
        self.L = input_dim
        self.D = 256  # Attention internal dim
        
        # Gated Attention Mechanism
        self.attention_V = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh()
        )
        self.attention_U = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Sigmoid()
        )
        self.attention_weights = nn.Linear(self.D, 1)
        
        # Bag-level Classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.L, 1)
        )
        
        # Instance-level Projection Head for Contrastive Learning
        self.projection_head = nn.Sequential(
            nn.Linear(self.L, self.L),
            nn.ReLU(),
            nn.Linear(self.L, projection_dim)
        )

    def forward(self, x, mask=None, return_projection=False):
        
        A_V = self.attention_V(x)  # [B, N, D]
        A_U = self.attention_U(x)  # [B, N, D]
        
        a_raw = self.attention_weights(A_V * A_U)  # [B, N, 1]
        
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1).to(a_raw.device)
            a_raw = a_raw.masked_fill(~mask_expanded, -1e9)
            
        A = torch.softmax(a_raw, dim=1)  # [B, N, 1]
        M = torch.sum(x * A, dim=1)  # [B, input_dim] - bag-level representation
        
        logits = self.classifier(M)  # [B, 1]
        
        if return_projection:
            Z = self.projection_head(x)  # [B, N, projection_dim]
            Z = F.normalize(Z, p=2, dim=-1)  # L2 normalization
            return logits, A, Z
        
        return logits, A


def bag_contrastive_loss(Z, A, mask, temperature=0.5, top_k_ratio=0.3):
    B, N, D = Z.shape
    A_squeezed = A.squeeze(-1)  # [B, N]
    
    total_loss = 0.0
    valid_bags = 0
    
    for b in range(B):
        # Obtain the valid instances of the current bag
        valid_mask = mask[b]  # [N]
        num_valid = valid_mask.sum().item()
        
        if num_valid <= 1:
            continue
            
        # Obtain effective features and attention weights
        Z_valid = Z[b][valid_mask]  # [num_valid, D]
        A_valid = A_squeezed[b][valid_mask]  # [num_valid]
        
        # Select the top-k instances with high attention as the key instances
        k = max(1, int(num_valid * top_k_ratio))
        top_k_indices = torch.topk(A_valid, k=min(k, num_valid))[1]
        
        # If there are fewer than 2 key instances, it is impossible to calculate the contrast loss.
        if len(top_k_indices) < 2:
            continue
        
        # Constructing positive sample pairs: Between key instances
        Z_key = Z_valid[top_k_indices]  # [k, D]
        
        # Calculate the similarity matrix between key instances
        sim_matrix = torch.matmul(Z_key, Z_key.T) / temperature  # [k, k]
        
        # Create mask: Exclude oneself
        mask_self = torch.eye(k, device=Z.device, dtype=torch.bool)
        
        # For each key instance, other key instances are regarded as positive samples.
        # All non-key instances are treated as negative samples
        Z_non_key = Z_valid[[i for i in range(num_valid) if i not in top_k_indices]]
        
        if len(Z_non_key) > 0:
            # Calculate the similarity between key instances and non-key instances
            sim_neg = torch.matmul(Z_key, Z_non_key.T) / temperature  # [k, num_non_key]
            
            # Calculate the loss for each key instance
            for i in range(k):
                # Positive sample: Other key example
                pos_sim = sim_matrix[i][~mask_self[i]]  # [k-1]
                
                # Negative sample: Non-critical instance
                neg_sim = sim_neg[i]  # [num_non_key]
                
                # InfoNCE-style loss
                # log(sum(exp(pos)) / (sum(exp(pos)) + sum(exp(neg))))
                pos_exp = torch.exp(pos_sim)
                neg_exp = torch.exp(neg_sim)
                
                denominator = pos_exp.sum() + neg_exp.sum()
                loss_i = -torch.log(pos_exp.sum() / (denominator + 1e-8) + 1e-8)
                
                total_loss += loss_i
        else:
            # If there are no non-critical instances, only comparisons will be made between the critical instances.
            for i in range(k):
                pos_sim = sim_matrix[i][~mask_self[i]]
                if len(pos_sim) > 0:
                    # Only use positive samples
                    loss_i = -torch.log(torch.exp(pos_sim).mean() + 1e-8)
                    total_loss += loss_i
        
        valid_bags += 1
    
    if valid_bags > 0:
        return total_loss / (valid_bags * k)  # The average loss of each key instance
    else:
        return torch.tensor(0.0, device=Z.device)