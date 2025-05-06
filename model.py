import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


class TextEncoder(nn.Module):
    """
    Bidirectional Transformer encoder for processing phoneme sequences.
    """
    def __init__(self, hidden_dim, n_layers=3, n_heads=8, dropout=0.1):
        super(TextEncoder, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.pos_encoder = PositionalEncoding(hidden_dim, dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=n_heads,
            dim_feedforward=hidden_dim*4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, n_layers)
        
    def forward(self, x, mask=None):
        """
        Args:
            x: Tensor, shape [batch_size, seq_length, hidden_dim]
            mask: Tensor, shape [batch_size, seq_length]
                
        Returns:
            output: Tensor, shape [batch_size, seq_length, hidden_dim]
        """
        x = self.pos_encoder(x)
        if mask is not None:
            # Convert mask to transformer format
            transformer_mask = mask.logical_not()
            output = self.transformer_encoder(x, src_key_padding_mask=transformer_mask)
        else:
            output = self.transformer_encoder(x)
        return output


class F0Encoder(nn.Module):
    """
    Convolutional encoder for processing F0 contours.
    """
    def __init__(self, hidden_dim, kernel_size=3, n_layers=5, dropout=0.1):
        super(F0Encoder, self).__init__()
        
        self.input_projection = nn.Linear(1, hidden_dim)
        
        # Stack of 1D convolutional layers with residual connections
        self.conv_layers = nn.ModuleList()
        for _ in range(n_layers):
            layer = nn.Sequential(
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding=(kernel_size - 1) // 2),
                nn.BatchNorm1d(hidden_dim),
                nn.LeakyReLU(0.1),
                nn.Dropout(dropout)
            )
            self.conv_layers.append(layer)
            
    def forward(self, f0):
        """
        Args:
            f0: Tensor, shape [batch_size, seq_length]
                
        Returns:
            output: Tensor, shape [batch_size, seq_length, hidden_dim]
        """
        # Add channel dimension
        x = f0.unsqueeze(-1)
        
        # Project to hidden dimension
        x = self.input_projection(x)
        
        # Apply convolutional layers with residual connections
        x_orig = x.transpose(1, 2)  # [batch, hidden_dim, seq_len]
        
        for layer in self.conv_layers:
            x_conv = layer(x_orig)
            x_orig = x_orig + x_conv  # Residual connection
            
        # Return to [batch, seq_len, hidden_dim] format
        output = x_orig.transpose(1, 2)
        
        return output


class Decoder(nn.Module):
    """
    Decoder that combines text and F0 information with style conditioning.
    Uses GRU layers with attention mechanism.
    """
    def __init__(self, hidden_dim, n_layers=2, dropout=0.1):
        super(Decoder, self).__init__()
        
        # Conditioning layers for style embedding
        self.style_condition_layers = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim*2),  # Scale and bias
        )
        
        # GRU layers
        self.gru_layers = nn.ModuleList()
        for _ in range(n_layers):
            self.gru_layers.append(nn.GRU(
                input_size=hidden_dim*2 if _ == 0 else hidden_dim,
                hidden_size=hidden_dim,
                batch_first=True,
                bidirectional=True
            ))
            
        # Output projection
        self.output_projection = nn.Linear(hidden_dim*2, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text_enc, f0_enc, style_embed):
        """
        Args:
            text_enc: Tensor, shape [batch_size, seq_length, hidden_dim]
            f0_enc: Tensor, shape [batch_size, seq_length, hidden_dim]
            style_embed: Tensor, shape [batch_size, hidden_dim]
                
        Returns:
            output: Tensor, shape [batch_size, seq_length, hidden_dim]
        """
        batch_size, seq_length, _ = text_enc.shape
        
        # Apply style conditioning
        style_cond = self.style_condition_layers(style_embed).view(batch_size, 1, 2, -1)
        style_scale, style_bias = style_cond[:, :, 0], style_cond[:, :, 1]
        
        # Apply FiLM conditioning on text encoding
        text_enc = text_enc * style_scale + style_bias
        
        # Concatenate text and F0 encoding
        decoder_input = torch.cat([text_enc, f0_enc], dim=-1)
        
        # Apply GRU layers
        x = decoder_input
        for gru in self.gru_layers:
            x, _ = gru(x)
            x = self.dropout(x)
            
        # Project to output dimension
        output = self.output_projection(x)
        
        return output


class HarmonicPredictor(nn.Module):
    """
    Predicts harmonic amplitudes for the DDSP synthesizer.
    """
    def __init__(self, hidden_dim, n_harmonics):
        super(HarmonicPredictor, self).__init__()
        
        self.n_harmonics = n_harmonics
        
        self.harmonic_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, n_harmonics),
            nn.Softmax(dim=-1)  # Ensure amplitudes sum to 1
        )
        
    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_length, hidden_dim]
                
        Returns:
            harmonic_amplitudes: Tensor, shape [batch_size, n_harmonics, seq_length]
        """
        batch_size, seq_length, _ = x.shape
        
        # Predict harmonic amplitudes
        amplitudes = self.harmonic_predictor(x)  # [batch, seq, n_harmonics]
        
        # Transpose to match DDSP input format
        harmonic_amplitudes = amplitudes.transpose(1, 2)  # [batch, n_harmonics, seq]
        
        return harmonic_amplitudes


class NoisePredictor(nn.Module):
    """
    Predicts noise band magnitudes for the DDSP synthesizer.
    """
    def __init__(self, hidden_dim, n_bands):
        super(NoisePredictor, self).__init__()
        
        self.n_bands = n_bands
        
        self.noise_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, n_bands),
            nn.Sigmoid()  # Ensure values are between 0 and 1
        )
        
    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_length, hidden_dim]
                
        Returns:
            noise_magnitudes: Tensor, shape [batch_size, n_bands, seq_length]
        """
        batch_size, seq_length, _ = x.shape
        
        # Predict noise magnitudes
        magnitudes = self.noise_predictor(x)  # [batch, seq, n_bands]
        
        # Transpose to match DDSP input format
        noise_magnitudes = magnitudes.transpose(1, 2)  # [batch, n_bands, seq]
        
        return noise_magnitudes


class TransientPredictor(nn.Module):
    """
    Predicts transient parameters for the DDSP synthesizer.
    Uses a combination of classification and regression approaches.
    """
    def __init__(self, hidden_dim, n_transients, max_transients):
        super(TransientPredictor, self).__init__()
        
        self.n_transients = n_transients
        self.max_transients = max_transients
        
        # Transient detection network
        self.transient_detector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # Probability of transient at each frame
        )
        
        # Transient parameter network (shared)
        self.transient_param_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.1)
        )
        
        # Transient ID prediction (classification)
        self.transient_id_predictor = nn.Linear(hidden_dim, n_transients)
        
        # Transient gain prediction
        self.transient_gain_predictor = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # Ensure gains are between 0 and 1
        )
        
    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_length, hidden_dim]
                
        Returns:
            dict containing:
                timings: Tensor, shape [batch_size, max_transients]
                ids: Tensor, shape [batch_size, max_transients]
                gains: Tensor, shape [batch_size, max_transients]
        """
        batch_size, seq_length, _ = x.shape
        device = x.device
        
        # Detect transient probabilities
        transient_probs = self.transient_detector(x).squeeze(-1)  # [batch, seq]
        
        # Process frame features
        frame_features = self.transient_param_net(x)  # [batch, seq, hidden]
        
        # Get transient IDs and gains for each frame
        transient_ids_logits = self.transient_id_predictor(frame_features)  # [batch, seq, n_transients]
        transient_gains_per_frame = self.transient_gain_predictor(frame_features).squeeze(-1)  # [batch, seq]
        
        # Initialize output tensors
        timings = torch.zeros(batch_size, self.max_transients, device=device)
        ids = torch.zeros(batch_size, self.max_transients, device=device, dtype=torch.long)
        gains = torch.zeros(batch_size, self.max_transients, device=device)
        
        # For each batch item, select the top max_transients frames with highest transient probability
        for b in range(batch_size):
            # Get top transient frames
            top_values, top_indices = torch.topk(transient_probs[b], min(self.max_transients, seq_length))
            
            # Only consider frames with probability > 0.5
            valid_indices = top_indices[top_values > 0.5]
            n_valid = len(valid_indices)
            
            if n_valid > 0:
                # Convert frame indices to timing values (assuming 10ms frames)
                timings[b, :n_valid] = valid_indices.float() * 0.01  # Convert to seconds
                
                # Get transient IDs (using argmax from logits)
                frame_ids = torch.argmax(transient_ids_logits[b, valid_indices], dim=-1)
                ids[b, :n_valid] = frame_ids
                
                # Get transient gains
                frame_gains = transient_gains_per_frame[b, valid_indices]
                gains[b, :n_valid] = frame_gains
        
        return {
            'timings': timings,
            'ids': ids,
            'gains': gains
        }


class ComponentGainPredictor(nn.Module):
    """
    Predicts mixing gains for the three DDSP components:
    harmonic, noise, and transient.
    """
    def __init__(self, hidden_dim):
        super(ComponentGainPredictor, self).__init__()
        
        self.gain_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, 3),
            nn.Softmax(dim=-1)  # Ensure gains sum to 1
        )
        
    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_length, hidden_dim]
                
        Returns:
            component_gains: Tensor, shape [batch_size, 3, seq_length]
        """
        # Predict component gains
        gains = self.gain_predictor(x)  # [batch, seq, 3]
        
        # Transpose to match DDSP input format
        component_gains = gains.transpose(1, 2)  # [batch, 3, seq]
        
        return component_gains


class PositionalEncoding(nn.Module):
    """
    Positional encoding for Transformer models.
    """
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_length, d_model]
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class DDSPSVS(nn.Module):
    """
    Combined Singing Voice Synthesis model using DDSP.
    """
    def __init__(self, 
                 n_phones,           # Vocabulary size
                 n_singers,          # Number of singers
                 n_languages,        # Number of languages
                 hidden_dim=512,     # Hidden dimension size
                 n_harmonics=100,    # Number of harmonics
                 n_noise_bands=80,   # Number of noise bands
                 n_transients=20,    # Number of transient templates
                 max_transients=8,   # Max transients per utterance
                 sample_rate=16000,  # Audio sample rate
                 ):
        super(DDSPSVS, self).__init__()
        
        # Embeddings
        self.phone_embedding = nn.Embedding(n_phones+1, hidden_dim)  # +1 for padding
        self.singer_embedding = nn.Embedding(n_singers, hidden_dim)
        self.language_embedding = nn.Embedding(n_languages, hidden_dim)
        
        # Encoders
        self.text_encoder = TextEncoder(hidden_dim)
        self.f0_encoder = F0Encoder(hidden_dim)
        
        # Decoder
        self.decoder = Decoder(hidden_dim)
        
        # Parameter predictors
        self.harmonic_predictor = HarmonicPredictor(hidden_dim, n_harmonics)
        self.noise_predictor = NoisePredictor(hidden_dim, n_noise_bands)
        self.transient_predictor = TransientPredictor(hidden_dim, n_transients, max_transients)
        self.component_gain_predictor = ComponentGainPredictor(hidden_dim)
        
        # Store configuration
        self.n_harmonics = n_harmonics
        self.n_noise_bands = n_noise_bands
        self.n_transients = n_transients
        self.max_transients = max_transients
        self.sample_rate = sample_rate
        
        # The DDSP synthesizer will be initialized during training
        # (it's kept separate since it might be loaded from a pretrained model)
        
    def forward(self, phone_seq, f0, singer_id, language_id, mask=None):
        """
        Args:
            phone_seq: Tensor, shape [batch_size, seq_length]
            f0: Tensor, shape [batch_size, seq_length]
            singer_id: Tensor, shape [batch_size]
            language_id: Tensor, shape [batch_size]
            mask: Tensor, shape [batch_size, seq_length], optional
                
        Returns:
            dict containing DDSP parameters:
                harmonic_amplitudes: [batch_size, n_harmonics, seq_length]
                noise_magnitudes: [batch_size, n_noise_bands, seq_length]
                transient_params: dict with timings, ids, gains
                component_gains: [batch_size, 3, seq_length]
        """
        # Embedding
        phone_embed = self.phone_embedding(phone_seq)
        singer_embed = self.singer_embedding(singer_id)
        language_embed = self.language_embedding(language_id)
        
        # Encoding
        text_enc = self.text_encoder(phone_embed, mask)
        f0_enc = self.f0_encoder(f0)
        
        # Combine singer and language embeddings
        style_embed = singer_embed + language_embed
        
        # Decoding
        decoder_output = self.decoder(text_enc, f0_enc, style_embed)
        
        # Predict DDSP parameters
        harmonic_amps = self.harmonic_predictor(decoder_output)
        noise_mags = self.noise_predictor(decoder_output)
        transient_params = self.transient_predictor(decoder_output)
        component_gains = self.component_gain_predictor(decoder_output)
        
        return {
            'harmonic_amplitudes': harmonic_amps,
            'noise_magnitudes': noise_mags,
            'transient_params': transient_params,
            'component_gains': component_gains
        }
