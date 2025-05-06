import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class TransientGenerator(nn.Module):
    """
    Generates transient sounds (e.g., consonant attacks) for DDSP synthesis.
    Uses a bank of pre-defined or learned templates that can be placed at specific times.
    """
    def __init__(self, sample_rate=16000, n_transients=20, template_size=1024):
        super(TransientGenerator, self).__init__()
        self.sample_rate = sample_rate
        self.n_transients = n_transients
        self.template_size = template_size
        
        # Either learn templates or initialize with sensible defaults
        self.init_templates()
        
    def init_templates(self):
        """
        Initialize transient templates.
        The templates are learned during training, but we start with reasonable defaults.
        """
        # Create learnable templates - each template represents a different type of transient
        self.templates = nn.Parameter(torch.zeros(self.n_transients, self.template_size))
        
        # Initialize with different types of transients
        for i in range(self.n_transients):
            # Different envelope characteristics for different templates
            attack_size = int(self.template_size * (0.1 + 0.4 * i / self.n_transients))
            decay_size = self.template_size - attack_size
            
            # Create attack-decay envelope
            attack = torch.linspace(0, 1, attack_size)
            decay = torch.exp(-torch.linspace(0, 5, decay_size))
            envelope = torch.cat([attack, decay])[:self.template_size]
            
            # Create different spectral characteristics
            noise = torch.randn(self.template_size)
            
            # For some templates, add resonances at different frequencies
            if i % 4 == 0:  # Every 4th template gets resonances
                freq1 = 100 + 200 * (i / self.n_transients)  # Hz
                freq2 = 2000 + 2000 * (i / self.n_transients)  # Hz
                
                # Create sinusoids with frequency-dependent envelope
                t = torch.linspace(0, self.template_size / self.sample_rate, self.template_size)
                sin1 = torch.sin(2 * math.pi * freq1 * t) * torch.exp(-t * 20)
                sin2 = torch.sin(2 * math.pi * freq2 * t) * torch.exp(-t * 40)
                
                # Combine with noise
                template = envelope * (noise * 0.5 + sin1 * 0.3 + sin2 * 0.2)
            else:
                # Pure noise with envelope
                template = envelope * noise
            
            # Normalize
            template = template / torch.max(torch.abs(template))
            
            # Assign to template bank
            self.templates.data[i] = template
    
    def forward(self, timings, ids, gains, audio_length=None):
        """
        Generate time-domain signals with transients placed at specified times.
        
        Args:
            timings: Timing values in seconds [batch_size, max_transients]
            ids: Template IDs for each transient [batch_size, max_transients]
            gains: Gain values for each transient [batch_size, max_transients]
            audio_length: Optional desired audio length (samples)
            
        Returns:
            transient_signal: Time-domain signal with transients [batch_size, n_samples]
        """
        batch_size, max_transients = timings.shape
        device = timings.device
        
        # Default audio length (3 seconds) if not provided
        if audio_length is None:
            audio_length = self.sample_rate * 3  # 3 seconds
        
        # Create output signal
        transient_signal = torch.zeros(batch_size, audio_length, device=device)
        
        # Place each transient in the output signal
        for b in range(batch_size):
            for t in range(max_transients):
                # Skip if gain is zero
                if gains[b, t] <= 0:
                    continue
                
                # Convert timing to sample position
                time_sec = timings[b, t].item()
                if time_sec <= 0:
                    continue  # Skip invalid timing
                    
                # Calculate sample position
                pos = int(time_sec * self.sample_rate)
                if pos >= audio_length:
                    continue  # Skip if out of bounds
                
                # Get template ID
                template_id = ids[b, t].item()
                if template_id >= self.n_transients:
                    template_id = self.n_transients - 1  # Clamp to valid range
                
                # Get template
                template = self.templates[template_id]
                
                # Calculate end position (with boundary check)
                end_pos = min(pos + len(template), audio_length)
                template_end = end_pos - pos
                
                # Place template in output signal
                transient_signal[b, pos:end_pos] += template[:template_end] * gains[b, t]
        
        # Normalize to prevent clipping
        max_vals = torch.max(torch.abs(transient_signal), dim=1, keepdim=True)[0]
        max_vals = torch.clamp(max_vals, min=1e-8)
        transient_signal = transient_signal / max_vals
        
        return transient_signal


class TransientLearner(nn.Module):
    """
    Extension module to learn transient templates from data.
    Can be used in place of hard-coded templates.
    """
    def __init__(self, sample_rate=16000, n_transients=20, template_size=1024, hidden_dim=512):
        super(TransientLearner, self).__init__()
        self.sample_rate = sample_rate
        self.n_transients = n_transients
        self.template_size = template_size
        
        # Network to generate templates
        self.template_generator = nn.Sequential(
            nn.Linear(n_transients, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, template_size),
            nn.Tanh()  # Keep values in [-1, 1] range
        )
    
    def forward(self, template_ids):
        """
        Generate transient templates based on IDs.
        
        Args:
            template_ids: One-hot encoded IDs [batch_size, n_transients]
            
        Returns:
            templates: Generated templates [batch_size, template_size]
        """
        return self.template_generator(template_ids)