import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt
import librosa
import librosa.display


def generate_harmonic_batch(instantaneous_phase, harmonic_numbers, phase, harmonic_amplitudes):
    """
    Function to generate harmonics efficiently without JIT compilation.
    """
    # Compute all harmonic phases at once
    harmonic_phases = instantaneous_phase.unsqueeze(1) * harmonic_numbers + phase.unsqueeze(-1)
    
    # Generate harmonic signals and apply amplitudes
    signals = torch.sin(harmonic_phases) * harmonic_amplitudes
    
    # Sum across harmonics
    output = torch.sum(signals, dim=1)
    
    return output


class HarmonicGenerator(nn.Module):
    """
    Fully optimized PyTorch module for generating harmonic content for singing voice synthesis.
    """
    
    def __init__(self, sample_rate=16000, n_harmonics=100):
        super(HarmonicGenerator, self).__init__()
        self.sample_rate = sample_rate
        self.n_harmonics = n_harmonics
        self.omega = 2 * np.pi / self.sample_rate  # Pre-compute constant
    
    def forward(self, f0, harmonic_amplitudes=None, phase=None):
        """
        Generate harmonic content based on fundamental frequency.
        
        Args:
            f0 (Tensor): Fundamental frequency contour [batch_size, time]
            harmonic_amplitudes (Tensor, optional): Amplitudes for each harmonic
                Shape [batch_size, n_harmonics, time] or [batch_size, time, n_harmonics]
            phase (Tensor, optional): Phase for each harmonic. If None, uses zero phase.
                Shape [batch_size, n_harmonics]
                
        Returns:
            Tensor: The generated harmonic signal [batch_size, time]
        """
        batch_size, n_frames = f0.shape
        device = f0.device
        
        # --- OPTIMIZATION: Use broadcasting instead of repeat for harmonic amplitudes ---
        if harmonic_amplitudes is None:
            harmonic_distribution = (1.0 / torch.arange(1, self.n_harmonics + 1, device=device)).unsqueeze(0).unsqueeze(-1)
            harmonic_amplitudes = harmonic_distribution.expand(batch_size, self.n_harmonics, n_frames)
        elif harmonic_amplitudes.dim() == 3 and harmonic_amplitudes.shape[1] == n_frames:
            harmonic_amplitudes = harmonic_amplitudes.transpose(1, 2)
        
        # --- OPTIMIZATION: Use in-place operations where possible ---
        harmonic_sum = harmonic_amplitudes.sum(dim=1, keepdim=True).clamp(min=1e-8)
        harmonic_amplitudes = harmonic_amplitudes / harmonic_sum
        
        # --- OPTIMIZATION: Pre-allocate phase tensor ---
        if phase is None:
            phase = torch.zeros(batch_size, self.n_harmonics, device=device)
        
        # --- OPTIMIZATION: Pre-compute time-related tensors once ---
        frame_time = torch.linspace(0, 1, n_frames, device=device)
        audio_length = int(n_frames * self.sample_rate / 100)  # Assuming 10ms frames
        sample_time = torch.linspace(0, 1, audio_length, device=device)
        
        # --- OPTIMIZATION: Improved vectorized F0 interpolation ---
        f0_interp = self._interpolate_f0_vectorized(f0, frame_time, sample_time, audio_length, device)
        
        # --- OPTIMIZATION: Compute phase with pre-computed omega constant ---
        f0_radians_per_sample = f0_interp * self.omega
        
        # --- OPTIMIZATION: Chunked cumsum for very long sequences ---
        chunk_size = min(10000, audio_length)
        instantaneous_phase = torch.zeros_like(f0_interp)
        
        for i in range(0, audio_length, chunk_size):
            end_idx = min(i + chunk_size, audio_length)
            if i == 0:
                instantaneous_phase[:, i:end_idx] = torch.cumsum(f0_radians_per_sample[:, i:end_idx], dim=1)
            else:
                # Continue cumsum from previous chunk's end value
                instantaneous_phase[:, i:end_idx] = (
                    instantaneous_phase[:, i-1].unsqueeze(1) + 
                    torch.cumsum(f0_radians_per_sample[:, i:end_idx], dim=1)
                )
        
        # --- OPTIMIZATION: Efficient interpolation of harmonic amplitudes ---
        harmonic_amplitudes_interp = F.interpolate(
            harmonic_amplitudes, 
            size=audio_length,
            mode='linear',
            align_corners=True
        )
        
        # --- OPTIMIZATION: Prepare harmonic numbers ---
        harmonic_numbers = torch.arange(1, self.n_harmonics + 1, device=device).view(1, -1, 1)
        
        # --- OPTIMIZATION: Process in chunks to manage memory ---
        signal = torch.zeros(batch_size, audio_length, device=device)
        chunk_size = min(8000, audio_length)
        
        for i in range(0, audio_length, chunk_size):
            end_idx = min(i + chunk_size, audio_length)
            
            # Process each chunk using the optimized function (without JIT)
            signal[:, i:end_idx] = generate_harmonic_batch(
                instantaneous_phase[:, i:end_idx],
                harmonic_numbers,
                phase,
                harmonic_amplitudes_interp[:, :, i:end_idx]
            )
        
        return signal
    
    def _interpolate_f0_vectorized(self, f0, frame_time, sample_time, audio_length, device):
        """
        Optimized vectorized f0 interpolation using PyTorch operations.
        """
        batch_size = f0.shape[0]
        f0_interp = torch.zeros(batch_size, audio_length, device=device)
        
        # --- OPTIMIZATION: Use boolean indexing and vectorized operations ---
        for b in range(batch_size):
            # Get voiced frames (where f0 > 0)
            voiced_mask = f0[b] > 0
            num_voiced = voiced_mask.sum().item()
            
            if num_voiced <= 0:
                continue  # No voiced frames
            elif num_voiced == 1:
                # If only one voiced frame, use constant value
                idx = voiced_mask.nonzero().item()
                f0_interp[b] = f0[b, idx]
            else:
                # --- OPTIMIZATION: Use vectorized interpolation ---
                voiced_frames = frame_time[voiced_mask]
                voiced_values = f0[b, voiced_mask]
                
                # Find indices of lower bounds for each sample time
                indices = torch.searchsorted(voiced_frames, sample_time)
                
                # Clamp indices to valid range
                indices = torch.clamp(indices, 1, len(voiced_frames)-1)
                
                # Get the two points for interpolation
                idx_low = indices - 1
                idx_high = indices
                
                # Get time points and values
                t_low = torch.gather(voiced_frames, 0, idx_low)
                t_high = torch.gather(voiced_frames, 0, idx_high)
                v_low = torch.gather(voiced_values, 0, idx_low)
                v_high = torch.gather(voiced_values, 0, idx_high)
                
                # Compute interpolation weights
                weights = (sample_time - t_low) / (t_high - t_low + 1e-7)  # Prevent division by zero
                
                # Perform linear interpolation
                f0_interp[b] = v_low + weights * (v_high - v_low)
        
        return f0_interp