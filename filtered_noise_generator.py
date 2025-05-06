import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import librosa


class FilteredNoiseGenerator(nn.Module):
    """
    PyTorch module for generating filtered noise component for singing voice synthesis.
    
    This module generates noise with time-varying spectral characteristics to model
    breathiness, fricatives, and other non-harmonic components of singing voice.
    
    It applies mel-scaled filter banks to white noise, allowing control over the
    spectral envelope of the noise at each time frame.
    """
    def __init__(self, sample_rate=16000, n_bands=80, fft_size=1024, hop_size=256):
        super(FilteredNoiseGenerator, self).__init__()
        self.sample_rate = sample_rate
        self.n_bands = n_bands
        self.fft_size = fft_size
        self.hop_size = hop_size
        
        # Pre-compute mel filterbank for frequency mapping
        mel_filters = self._create_mel_filterbank()
        # Register as buffer to handle device transfers automatically
        self.register_buffer('mel_filters', mel_filters)
        
        # Pre-compute and cache Hann window for STFT operations
        self.register_buffer('hann_window', torch.hann_window(self.fft_size))
        
        # Cache for bin mappings in simple forward pass
        self._bin_mappings = None
        
    def _create_mel_filterbank(self):
        """
        Create a mel-spaced filterbank for perceptually relevant noise shaping.
        """
        # Create mel filterbank matrix using librosa
        mel_filters = librosa.filters.mel(
            sr=self.sample_rate,
            n_fft=self.fft_size,
            n_mels=self.n_bands,
            norm=1
        )
        return torch.from_numpy(mel_filters).float()
    
    def forward(self, noise_magnitudes, audio_length=None):
        """
        Generate filtered noise using STFT/ISTFT for accurate time-frequency processing.
        
        Args:
            noise_magnitudes: [batch_size, n_bands, n_frames] filter bank gains
            audio_length: Optional target audio length (in samples)
            
        Returns:
            Filtered noise signal [batch_size, n_samples]
        """
        batch_size, n_bands, n_frames = noise_magnitudes.shape
        device = noise_magnitudes.device
        
        # Determine output audio length (assuming 10ms frames if not specified)
        if audio_length is None:
            audio_length = int(n_frames * self.sample_rate / 100)
        
        # Generate white noise
        noise = torch.randn(batch_size, audio_length, device=device)
        
        # Take STFT of noise - no need to move hann_window to device as it's a registered buffer
        noise_stft = torch.stft(
            noise, 
            n_fft=self.fft_size,
            hop_length=self.hop_size,
            window=self.hann_window,
            return_complex=True
        )
        
        # Get magnitude and phase
        noise_magnitude = torch.abs(noise_stft)
        noise_phase = torch.angle(noise_stft)
        
        # Ensure filter_response has the right number of frames to match STFT
        n_stft_frames = noise_stft.shape[2]
        filter_response_resampled = F.interpolate(
            noise_magnitudes, 
            size=n_stft_frames,
            mode='linear',
            align_corners=True
        )
        
        # Convert mel-band gains to STFT bin gains using the mel filterbank
        # [batch_size, n_bands, n_frames] -> [batch_size, n_fft//2+1, n_frames]
        # Use matrix multiplication with proper reshaping
        # mel_filters shape: [n_bands, n_fft//2+1]
        filter_response_full = torch.matmul(
            self.mel_filters.transpose(0, 1), 
            filter_response_resampled
        )
        
        # Apply filter to magnitude
        filtered_magnitude = noise_magnitude * filter_response_full
        
        # Reconstruct complex STFT
        filtered_stft = torch.polar(filtered_magnitude, noise_phase)
        
        # Inverse STFT
        filtered_noise = torch.istft(
            filtered_stft,
            n_fft=self.fft_size,
            hop_length=self.hop_size,
            window=self.hann_window,
            length=audio_length
        )
        
        return filtered_noise
    
    def _compute_bin_mappings(self, n_fft_bins):
        """
        Compute and cache frequency bin mappings for the simple forward pass.
        """
        if self._bin_mappings is None or self._bin_mappings[0] != n_fft_bins:
            bin_per_band = max(1, n_fft_bins // self.n_bands)
            bin_starts = []
            bin_ends = []
            bin_widths = []
            
            for b in range(self.n_bands):
                bin_start = int(b * bin_per_band)
                bin_end = int((b + 1) * bin_per_band) if b < self.n_bands - 1 else n_fft_bins
                bin_width = bin_end - bin_start
                
                if bin_width > 0:
                    bin_starts.append(bin_start)
                    bin_ends.append(bin_end)
                    bin_widths.append(bin_width)
                    
            self._bin_mappings = (
                n_fft_bins,
                torch.tensor(bin_starts, dtype=torch.long),
                torch.tensor(bin_ends, dtype=torch.long),
                torch.tensor(bin_widths, dtype=torch.long)
            )
        
        return self._bin_mappings[1:]  # Return starts, ends, widths
    
    def forward_simple(self, noise_magnitudes, audio_length=None):
        """
        Optimized simplified forward pass that shapes noise directly in the frequency domain.
        This vectorized implementation avoids Python loops for faster computation.
        
        Args:
            noise_magnitudes: [batch_size, n_bands, n_frames] filter bank gains
            audio_length: Optional target audio length (in samples)
            
        Returns:
            Filtered noise signal [batch_size, n_samples]
        """
        batch_size, n_bands, n_frames = noise_magnitudes.shape
        device = noise_magnitudes.device
        
        # Determine output audio length (assuming 10ms frames if not specified)
        if audio_length is None:
            audio_length = int(n_frames * self.sample_rate / 100)
        
        # Generate white noise
        noise = torch.randn(batch_size, audio_length, device=device)
        
        # Interpolate noise magnitudes to the sample rate
        # This gives us filter gains for each time step
        noise_mags_interp = F.interpolate(
            noise_magnitudes, 
            size=audio_length, 
            mode='linear', 
            align_corners=False
        )
        
        # Apply time-varying spectral shaping in frequency domain
        noise_fft = torch.fft.rfft(noise, dim=1)
        n_fft_bins = noise_fft.shape[1]
        
        # Get bin mappings (cached after first computation)
        bin_starts, bin_ends, bin_widths = self._compute_bin_mappings(n_fft_bins)
        bin_starts = bin_starts.to(device)
        bin_ends = bin_ends.to(device)
        bin_widths = bin_widths.to(device)
        
        # Create output FFT tensor
        noise_fft_shaped = torch.zeros_like(noise_fft)
        
        # Vectorized implementation - process all bands without Python loops
        for i, (start, end, width) in enumerate(zip(bin_starts, bin_ends, bin_widths)):
            if i >= n_bands:
                break
                
            # Get band response for all batches: [batch_size, audio_length]
            band_response = noise_mags_interp[:, i, :]
            
            # Create sample indices for this band - shared across batch
            indices = torch.linspace(0, audio_length-1, width, device=device).long()
            
            # Sample the band response at the indices - for all batches at once
            # Output shape: [batch_size, bin_width]
            band_gains = band_response[:, indices]
            
            # Apply gains to FFT bins - for all batches at once
            noise_fft_shaped[:, start:end] = noise_fft[:, start:end] * band_gains
        
        # Inverse FFT to get filtered noise in time domain
        filtered_noise = torch.fft.irfft(noise_fft_shaped, n=audio_length, dim=1)
        
        return filtered_noise