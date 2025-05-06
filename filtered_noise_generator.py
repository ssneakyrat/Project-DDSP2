import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchaudio


class FilteredNoiseGenerator(nn.Module):
    """
    Generates filtered noise for DDSP synthesis.
    Uses a multiband filterbank to shape noise according to desired spectral characteristics.
    """
    def __init__(self, sample_rate=16000, n_bands=80):
        super(FilteredNoiseGenerator, self).__init__()
        self.sample_rate = sample_rate
        self.n_bands = n_bands
        
        # Initialize filter bank parameters
        self.init_filter_bank()
        
    def init_filter_bank(self):
        """
        Initialize the filterbank for noise shaping.
        Uses mel-scale band edges for perceptually relevant filtering.
        """
        # Compute mel-scale filter bank centers
        mel_min = 0
        mel_max = torchaudio.transforms.MelScale()(torch.tensor([self.sample_rate / 2])).item()
        
        # Compute band edges in mel scale
        mel_edges = torch.linspace(mel_min, mel_max, self.n_bands + 1)
        
        # Convert mel band edges to frequency
        hz_edges = torchaudio.functional.mel_to_hz(mel_edges)
        
        # Normalize frequency to Nyquist (for torch.stft compatibility)
        self.band_edges = hz_edges / (self.sample_rate / 2)
        
    def get_noise_magnitudes_mask(self, magnitudes, fft_size):
        """
        Create frequency masks from noise magnitude controls.
        
        Args:
            magnitudes: Control values for each frequency band [batch_size, n_bands, n_frames]
            fft_size: Size of FFT to use for filtering
            
        Returns:
            mask: Frequency domain mask for noise shaping [batch_size, n_frames, fft_size//2 + 1]
        """
        batch_size, n_bands, n_frames = magnitudes.shape
        device = magnitudes.device
        
        # Create mask for each frequency bin
        n_freqs = fft_size // 2 + 1
        mask = torch.zeros(batch_size, n_frames, n_freqs, device=device)
        
        # Map frequency bins to band indices
        freq_indices = torch.linspace(0, 1, n_freqs, device=device)
        
        # Fill the mask based on band magnitudes
        for band in range(self.n_bands):
            # Find frequency bins that fall within this band
            band_start = self.band_edges[band].item()
            band_end = self.band_edges[band + 1].item()
            
            # Create binary mask for this band
            band_mask = (freq_indices >= band_start) & (freq_indices < band_end)
            
            # Apply band magnitude to all frequencies in this band
            for b in range(batch_size):
                for f in range(n_frames):
                    mask[b, f, band_mask] = magnitudes[b, band, f]
        
        return mask
    
    def forward(self, magnitudes, audio_length=None):
        """
        Generate time-domain filtered noise based on magnitude controls.
        
        Args:
            magnitudes: Filter gains for each band [batch_size, n_bands, n_frames]
            audio_length: Optional desired audio length (samples)
            
        Returns:
            filtered_noise: Time-domain filtered noise signal [batch_size, n_samples]
        """
        batch_size, n_bands, n_frames = magnitudes.shape
        device = magnitudes.device
        
        # Calculate expected audio length (assuming 10ms frames) if not provided
        if audio_length is None:
            audio_length = int(n_frames * self.sample_rate / 100)  # For 10ms frames
        
        # Generate white noise
        white_noise = torch.randn(batch_size, audio_length, device=device)
        
        # Calculate STFT parameters
        fft_size = min(512, audio_length)  # Reasonable FFT size
        hop_length = fft_size // 4
        win_length = fft_size
        
        # Apply STFT
        noise_stft = torch.stft(
            white_noise,
            n_fft=fft_size,
            hop_length=hop_length,
            win_length=win_length,
            window=torch.hann_window(win_length, device=device),
            return_complex=True
        )
        
        # Calculate frequency mask from magnitudes
        # First, interpolate magnitudes to match STFT frames
        stft_frames = noise_stft.shape[1]
        magnitudes_interp = F.interpolate(
            magnitudes, 
            size=stft_frames,
            mode='linear',
            align_corners=True
        )
        
        # Create mask
        mask = self.get_noise_magnitudes_mask(magnitudes_interp, fft_size)
        mask = mask[:, :stft_frames, :]
        
        # Apply mask to STFT
        filtered_stft = noise_stft * mask.unsqueeze(-1)
        
        # Convert back to time domain
        filtered_noise = torch.istft(
            filtered_stft,
            n_fft=fft_size,
            hop_length=hop_length,
            win_length=win_length,
            window=torch.hann_window(win_length, device=device),
            length=audio_length
        )
        
        # Normalize audio to prevent clipping
        filtered_noise = filtered_noise / (torch.max(torch.abs(filtered_noise), dim=1, keepdim=True)[0] + 1e-8)
        
        return filtered_noise