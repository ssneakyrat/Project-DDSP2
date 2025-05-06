import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class HarmonicGenerator(nn.Module):
    """
    Generates harmonic signals based on fundamental frequency (F0) and harmonic amplitudes.
    Uses efficient cumulative phase modeling for high-quality sinusoidal synthesis.
    """
    def __init__(self, sample_rate=16000, n_harmonics=100):
        super(HarmonicGenerator, self).__init__()
        self.sample_rate = sample_rate
        self.n_harmonics = n_harmonics
        
    def forward(self, f0, amplitudes, initial_phase=None):
        """
        Generate time-domain harmonic signals based on F0 and harmonic amplitudes.
        
        Args:
            f0: Fundamental frequency contour [batch_size, n_frames]
            amplitudes: Harmonic amplitudes [batch_size, n_harmonics, n_frames]
            initial_phase: Optional initial phase for each harmonic [batch_size, n_harmonics]
            
        Returns:
            harmonic_signal: Time-domain harmonic signal [batch_size, n_samples]
        """
        batch_size, n_frames = f0.shape
        device = f0.device
        
        # Ensure harmonics dimension matches n_harmonics
        if amplitudes.shape[1] != self.n_harmonics:
            raise ValueError(f"Expected {self.n_harmonics} harmonics but got {amplitudes.shape[1]}")
        
        # Calculate expected audio length (assuming 10ms frames)
        n_samples = int(n_frames * self.sample_rate / 100)  # For 10ms frames
        
        # Create harmonic frequencies (harmonic multipliers for f0)
        harmonic_numbers = torch.arange(1, self.n_harmonics + 1, device=device).float()
        
        # Upsample F0 and amplitudes to sample rate using linear interpolation
        # First create the original time points (frame indices)
        original_time = torch.linspace(0, 1, n_frames, device=device)
        
        # Then create time points for the output samples
        sample_time = torch.linspace(0, 1, n_samples, device=device)
        
        # Interpolate F0
        f0_upsampled = F.interpolate(
            f0.unsqueeze(1), 
            size=n_samples,
            mode='linear',
            align_corners=True
        ).squeeze(1)
        
        # Interpolate amplitudes
        amplitudes_upsampled = F.interpolate(
            amplitudes, 
            size=n_samples,
            mode='linear',
            align_corners=True
        )
        
        # Create harmonic signal
        harmonic_signal = torch.zeros(batch_size, n_samples, device=device)
        
        # Use cumulative phase for accurate synthesis
        # Create time steps
        time_step = 1.0 / self.sample_rate
        
        # Calculate phase increment per sample for each frequency
        phase_increment = 2 * math.pi * f0_upsampled.unsqueeze(-1) * harmonic_numbers.unsqueeze(0)
        phase_increment = phase_increment.transpose(1, 2)  # [batch, n_harmonics, n_samples]
        
        # Default initial phase
        if initial_phase is None:
            initial_phase = torch.zeros(batch_size, self.n_harmonics, device=device)
            
        # Calculate cumulative phase
        phases = torch.zeros(batch_size, self.n_harmonics, n_samples, device=device)
        
        # Set initial phase
        current_phase = initial_phase.unsqueeze(-1)  # [batch, n_harmonics, 1]
        
        # Faster cumulative phase calculation (vectorized)
        # Phase is the cumulative sum of phase increments
        # phase_increments: [batch, n_harmonics, n_samples]
        phases = torch.cumsum(phase_increment * time_step, dim=2)
        
        # Add initial phase
        phases = phases + current_phase
        
        # Generate sinusoids and apply amplitudes
        # sin_waves: [batch, n_harmonics, n_samples]
        sin_waves = torch.sin(phases)
        
        # Apply amplitudes and sum all harmonics
        harmonic_signals = sin_waves * amplitudes_upsampled
        harmonic_signal = torch.sum(harmonic_signals, dim=1)
        
        # Handle NaN and Inf values (from zero or negative F0)
        harmonic_signal = torch.nan_to_num(harmonic_signal, nan=0.0, posinf=0.0, neginf=0.0)
        
        return harmonic_signal