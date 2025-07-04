"""
降噪神經網絡模型定義
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FrequencyTracking(nn.Module):
    def __init__(self, freq_dim, hidden_dim, dropout=0.2):
        super().__init__()
        self.freq_attention = nn.Sequential(
            nn.Linear(freq_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, freq_dim),
            nn.Softmax(dim=-1)
        )
        self.freq_gate = nn.Sequential(
            nn.Linear(freq_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, freq_dim),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        attention_weights = self.freq_attention(x)
        gate_weights = self.freq_gate(x)
        return x * attention_weights * gate_weights


class BurstDetector(nn.Module):
    def __init__(self, input_dim, threshold=0.6):
        super().__init__()
        self.detector = nn.Sequential(
            nn.Linear(input_dim, input_dim//2),
            nn.ReLU(),
            nn.Linear(input_dim//2, input_dim),
            nn.Sigmoid()
        )
        self.threshold = threshold
        
    def forward(self, x):
        burst_prob = self.detector(x)
        burst_mask = (burst_prob > self.threshold).float()
        return burst_prob, burst_mask


class AdaptiveNotchFilter(nn.Module):
    def __init__(self, freq_dim, num_filters=4):
        super().__init__()
        self.num_filters = num_filters
        self.freq_dim = freq_dim
        self.freq_estimator = nn.Sequential(
            nn.Linear(freq_dim, freq_dim//2),
            nn.ReLU(),
            nn.Linear(freq_dim//2, num_filters * 2)
        )
        
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        params = self.freq_estimator(x)
        center_freqs = params[:, :, :self.num_filters]
        bandwidths = F.softplus(params[:, :, self.num_filters:]) * 1.3
        
        freq_idx = torch.linspace(0, 1, self.freq_dim).to(x.device)
        freq_idx = freq_idx.view(1, 1, -1)
        
        low_freq_protection = (freq_idx > 0.1).float()
        
        output = x
        for i in range(self.num_filters):
            center = center_freqs[:, :, i:i+1]
            bandwidth = bandwidths[:, :, i:i+1]
            notch_response = 1 - torch.exp(-((freq_idx - center) ** 2) / (2 * bandwidth ** 2))
            notch_response = notch_response * low_freq_protection + 1.0 * (1 - low_freq_protection)
            output = output * notch_response
            
        return output


class DFSMNLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, memory_size=50, look_ahead=5, stride=1, dropout=0.2):
        super().__init__()
        self.memory_size = memory_size
        self.look_ahead = look_ahead
        self.stride = stride
        
        self.linear = nn.Linear(input_dim, hidden_dim)
        self.memory_weights = nn.Parameter(torch.randn(memory_size, hidden_dim))
        self.look_ahead_weights = nn.Parameter(torch.randn(look_ahead, hidden_dim))
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        h = self.linear(x)
        h = self.dropout(h)
        output = torch.zeros_like(h)
        
        for t in range(seq_len):
            current = h[:, t]
            
            memory_start = max(0, t - self.memory_size * self.stride)
            memory_end = t
            if memory_start < memory_end:
                memory_states = h[:, memory_start:memory_end:self.stride]
                actual_memory_steps = memory_states.size(1)
                memory_weights_used = self.memory_weights[:actual_memory_steps]
                memory_context = torch.matmul(
                    memory_states.transpose(1, 2),
                    memory_weights_used
                ).sum(dim=-1)
                current = current + memory_context
            
            future_start = t + self.stride
            future_end = min(seq_len, t + (self.look_ahead * self.stride) + 1)
            if future_start < future_end:
                future_states = h[:, future_start:future_end:self.stride]
                actual_future_steps = future_states.size(1)
                future_weights_used = self.look_ahead_weights[:actual_future_steps]
                future_context = torch.matmul(
                    future_states.transpose(1, 2),
                    future_weights_used
                ).sum(dim=-1)
                current = current + future_context
            
            output[:, t] = current
        
        output = self.layer_norm(output)
        return output


class EnhancedDFSMN(nn.Module):
    def __init__(self, input_dim, hidden_dims=[512, 384, 256, 128], output_dim=257, dropout=0.2):
        super().__init__()
        
        self.dfsmn_layers = nn.ModuleList()
        self.dfsmn_layers.append(DFSMNLayer(input_dim, hidden_dims[0], dropout=dropout))
        for i in range(1, len(hidden_dims)):
            self.dfsmn_layers.append(DFSMNLayer(hidden_dims[i-1], hidden_dims[i], dropout=dropout))
        
        self.freq_tracking = FrequencyTracking(input_dim, hidden_dims[0], dropout=dropout)
        self.burst_detector = BurstDetector(input_dim)
        self.notch_filter = AdaptiveNotchFilter(input_dim)
        
        self.skip_connections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dims[i]),
                nn.Dropout(dropout)
            ) for i in range(len(hidden_dims))
        ])
        
        self.output = nn.Linear(hidden_dims[-1], output_dim)
        self.output_gate = nn.Sequential(
            nn.Linear(hidden_dims[-1], output_dim),
            nn.Dropout(dropout),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        freq_features = self.freq_tracking(x)
        notch_filtered = self.notch_filter(freq_features)
        
        burst_prob, burst_mask = self.burst_detector(x)
        
        skip_features = x
        features = notch_filtered
        for i, layer in enumerate(self.dfsmn_layers):
            skip = self.skip_connections[i](skip_features)
            features = layer(features)
            features = F.relu(features + skip)
        
        output = self.output(features)
        gate = self.output_gate(features)
        
        output = output * (1 - burst_mask) + x * burst_mask
        final_output = output * gate
        
        return final_output, burst_prob