import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class TransientGenerator(nn.Module):
    """
    Optimized PyTorch module for generating transient sounds like consonants, plosives and glottal stops.
    Uses a learned dictionary of templates for efficient transient sound generation.
    """
    def __init__(self, sample_rate=16000, n_transients=20, transient_duration=0.1):
        super(TransientGenerator, self).__init__()
        self.sample_rate = sample_rate
        self.n_transients = n_transients
        self.transient_samples = int(transient_duration * sample_rate)
        
        # Create a dictionary of transient templates
        # In training, these would be learned from data
        self.transient_templates = nn.Parameter(
            torch.randn(n_transients, self.transient_samples),
            requires_grad=True
        )
        
        # Initialize with realistic transient shapes (exponential decay)
        with torch.no_grad():
            time = torch.linspace(0, transient_duration, self.transient_samples)
            decay_rates = torch.linspace(5, 50, n_transients).view(-1, 1)
            envelopes = torch.exp(-decay_rates * time)
            
            # Create different spectral characteristics for each transient
            frequencies = torch.linspace(50, 5000, n_transients).view(-1, 1)
            sinusoids = torch.sin(2 * np.pi * frequencies * time)
            
            # Apply envelopes to sinusoids and add noise
            templates = envelopes * sinusoids
            templates += 0.3 * torch.randn_like(templates) * envelopes
            
            # Normalize templates
            templates = templates / templates.abs().max(dim=1, keepdim=True)[0]
            
            self.transient_templates.data = templates
    
    def forward(self, transient_timings, transient_ids, transient_gains, audio_length=None):
        """
        Generate transient sounds at specific times using efficient placement algorithm.
        
        Args:
            transient_timings: [batch_size, max_transients] timings in seconds
            transient_ids: [batch_size, max_transients] indices into template dictionary
            transient_gains: [batch_size, max_transients] amplitude for each transient
            audio_length: Optional target audio length (in samples)
            
        Returns:
            Transient signal [batch_size, n_samples]
        """
        batch_size = transient_timings.shape[0]
        device = transient_timings.device
        max_transients = transient_timings.shape[1]
        
        # Determine output audio length
        if audio_length is None:
            # Calculate total audio length based on the latest transient
            max_time = torch.max(transient_timings).item()
            audio_length = int((max_time + 0.5) * self.sample_rate)  # Add buffer
        
        # Initialize output signal
        signal = torch.zeros(batch_size, audio_length, device=device)
        
        # Get transient templates for the requested IDs
        # This is much more efficient than individually placing each transient
        for batch_idx in range(batch_size):
            for t in range(max_transients):
                # Skip if gain is zero (no transient)
                if transient_gains[batch_idx, t] <= 0:
                    continue
                
                # Get transient ID and convert timing to sample index
                t_id = transient_ids[batch_idx, t].long()
                if t_id >= self.n_transients:
                    continue  # Skip invalid IDs
                
                t_sample = int(transient_timings[batch_idx, t] * self.sample_rate)
                
                # Skip if transient would be outside the audio bounds
                if t_sample >= audio_length:
                    continue
                
                # Get the transient template
                template = self.transient_templates[t_id]
                
                # Calculate valid range for placement
                template_length = template.shape[0]
                end_sample = min(t_sample + template_length, audio_length)
                valid_length = end_sample - t_sample
                
                # Place the transient in the signal
                signal[batch_idx, t_sample:end_sample] += (
                    template[:valid_length] * transient_gains[batch_idx, t]
                )
        
        return signal
    
    def generate_random_transients(self, batch_size, n_transients_per_batch=5, max_time=3.0):
        """
        Helper method to generate random transient events for testing.
        
        Args:
            batch_size: Number of examples in batch
            n_transients_per_batch: Number of transients to generate per example
            max_time: Maximum audio length in seconds
            
        Returns:
            Tuple of (transient_timings, transient_ids, transient_gains)
        """
        device = self.transient_templates.device
        
        # Random timings between 0 and max_time
        transient_timings = torch.rand(batch_size, n_transients_per_batch, device=device) * max_time
        
        # Sort timings for more realistic timing
        transient_timings, _ = torch.sort(transient_timings, dim=1)
        
        # Random transient IDs
        transient_ids = torch.randint(0, self.n_transients, 
                                     (batch_size, n_transients_per_batch),
                                     device=device)
        
        # Random gains between 0.1 and 1.0
        transient_gains = 0.1 + 0.9 * torch.rand(batch_size, n_transients_per_batch, device=device)
        
        return transient_timings, transient_ids, transient_gains