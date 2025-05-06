import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import librosa
import librosa.display
from harmonic_generator import HarmonicGenerator
from filtered_noise_generator import FilteredNoiseGenerator
from transient_generator import TransientGenerator


class DDSPSynthesizer(nn.Module):
    """
    Complete DDSP synthesizer for singing voice that combines harmonic, noise, and transient generators.
    """
    def __init__(self, sample_rate=16000, n_harmonics=100, n_noise_bands=80, n_transients=20):
        super(DDSPSynthesizer, self).__init__()
        self.sample_rate = sample_rate
        
        # Initialize the three core components
        self.harmonic_generator = HarmonicGenerator(sample_rate=sample_rate, n_harmonics=n_harmonics)
        self.noise_generator = FilteredNoiseGenerator(sample_rate=sample_rate, n_bands=n_noise_bands)
        self.transient_generator = TransientGenerator(sample_rate=sample_rate, n_transients=n_transients)
        
    def forward(self, 
               f0,                    # [batch_size, n_frames]
               harmonic_amplitudes,   # [batch_size, n_harmonics, n_frames]
               harmonic_phase=None,   # [batch_size, n_harmonics]
               noise_magnitudes=None, # [batch_size, n_noise_bands, n_frames]
               transient_timings=None, # [batch_size, max_transients]
               transient_ids=None,     # [batch_size, max_transients]
               transient_gains=None,   # [batch_size, max_transients]
               component_gains=None    # [batch_size, 3, n_frames] for harmonic, noise, transient
              ):
        """
        Generate a singing voice signal by combining harmonic, noise, and transient components.
        
        Args:
            f0: Fundamental frequency [batch_size, n_frames]
            harmonic_amplitudes: Amplitudes for each harmonic [batch_size, n_harmonics, n_frames]
            harmonic_phase: Optional phase for harmonics [batch_size, n_harmonics]
            noise_magnitudes: Filter gains for noise [batch_size, n_noise_bands, n_frames]
            transient_timings: Timings for transients [batch_size, max_transients]
            transient_ids: Template IDs for transients [batch_size, max_transients]
            transient_gains: Amplitudes for transients [batch_size, max_transients]
            component_gains: Mixing gains for components [batch_size, 3, n_frames]
            
        Returns:
            Complete synthesized audio signal [batch_size, n_samples]
        """
        batch_size, n_frames = f0.shape
        device = f0.device
        
        # Calculate expected audio length
        audio_length = int(n_frames * self.sample_rate / 100)  # Assuming 10ms frames
        
        # Generate harmonic component
        harmonic_signal = self.harmonic_generator(f0, harmonic_amplitudes, harmonic_phase)
        
        # Generate noise component if provided
        if noise_magnitudes is not None:
            noise_signal = self.noise_generator(noise_magnitudes, audio_length=audio_length)
        else:
            noise_signal = torch.zeros_like(harmonic_signal)
            
        # Generate transient component if provided
        if transient_timings is not None and transient_ids is not None and transient_gains is not None:
            transient_signal = self.transient_generator(
                transient_timings, transient_ids, transient_gains, audio_length=audio_length
            )
        else:
            transient_signal = torch.zeros_like(harmonic_signal)
        
        # Default component gains if not provided
        if component_gains is None:
            component_gains = torch.ones(batch_size, 3, n_frames, device=device)
            # Default mix: 70% harmonic, 20% noise, 10% transient
            component_gains[:, 0] *= 0.7  # harmonic
            component_gains[:, 1] *= 0.2  # noise
            component_gains[:, 2] *= 0.1  # transient
        
        # Interpolate component gains to sample rate
        gains_interp = F.interpolate(
            component_gains, 
            size=audio_length,
            mode='linear',
            align_corners=True
        )
        
        # Mix the components with their respective gains
        output_signal = (
            harmonic_signal * gains_interp[:, 0] + 
            noise_signal * gains_interp[:, 1] + 
            transient_signal * gains_interp[:, 2]
        )
        
        # Normalize output to prevent clipping
        output_signal = output_signal / torch.max(
            torch.abs(output_signal).max(dim=1)[0], 
            torch.tensor(1e-8, device=device)
        ).unsqueeze(1)
        
        return output_signal


def generate_test_singing_parameters(batch_size=1, 
                                    duration_seconds=3.0, 
                                    sample_rate=16000,
                                    n_harmonics=40, 
                                    n_noise_bands=80,
                                    n_transients=20,
                                    max_transients=8,
                                    device='cpu'):
    """
    Generate test parameters for synthesizing singing voice.
    """
    # Calculate number of frames (assuming 10ms per frame)
    n_frames = int(duration_seconds * 100)
    
    # Create a melody contour with vibrato
    base_f0 = torch.zeros(batch_size, n_frames, device=device)
    
    # Define note pitches in Hz (A3, C4, E4, G4, A4)
    note_freqs = torch.tensor([220.0, 261.63, 329.63, 392.0, 440.0], device=device)
    
    # Create a simple melody pattern
    note_pattern = [0, 1, 2, 3, 4, 3, 2, 1]
    note_duration = n_frames // (len(note_pattern) * 2)  # Each note gets same duration
    
    for b in range(batch_size):
        # Fill in the f0 contour with the melody
        for i, note_idx in enumerate(note_pattern):
            start_frame = i * note_duration
            end_frame = start_frame + note_duration
            if end_frame <= n_frames:
                base_f0[b, start_frame:end_frame] = note_freqs[note_idx]
    
    # Add vibrato (5 Hz)
    vibrato_rate = 5.0
    vibrato_depth = 20.0  # Hz
    time_idx = torch.linspace(0, duration_seconds, n_frames, device=device)
    vibrato = vibrato_depth * torch.sin(2 * np.pi * vibrato_rate * time_idx)
    
    # Apply vibrato only to longer notes (smooth transition between notes)
    vibrato_mask = torch.ones_like(base_f0)
    for i in range(1, len(note_pattern)):
        transition_frames = 5  # frames for transition
        start_frame = i * note_duration - transition_frames
        end_frame = i * note_duration + transition_frames
        if 0 <= start_frame < n_frames and end_frame < n_frames:
            vibrato_mask[:, start_frame:end_frame] = 0.0
    
    f0 = base_f0 + vibrato.unsqueeze(0) * vibrato_mask
    
    # Create time-varying harmonic amplitudes (formant-like)
    harmonic_indexes = torch.arange(1, n_harmonics + 1, device=device).float()
    
    # Create moving formants
    formant1_freq = torch.linspace(600, 800, n_frames, device=device)  # Hz
    formant2_freq = torch.linspace(1200, 900, n_frames, device=device)  # Hz
    formant1_bw = 100.0  # Bandwidth in Hz
    formant2_bw = 120.0  # Bandwidth in Hz
    
    # Initialize harmonic amplitudes
    harmonic_amplitudes = torch.zeros(batch_size, n_harmonics, n_frames, device=device)
    
    # Create formant structure that varies over time
    for b in range(batch_size):
        for t in range(n_frames):
            current_f0 = f0[b, t].item()
            if current_f0 > 0:  # Only for voiced frames
                harmonic_freqs = harmonic_indexes * current_f0
                
                # Formant filter responses
                formant1_response = torch.exp(-((harmonic_freqs - formant1_freq[t])**2) / (2 * formant1_bw**2))
                formant2_response = torch.exp(-((harmonic_freqs - formant2_freq[t])**2) / (2 * formant2_bw**2))
                
                # Combine formants with spectral tilt
                spectral_tilt = 1.0 / harmonic_indexes
                harmonic_amplitudes[b, :, t] = (formant1_response + 0.7 * formant2_response) * spectral_tilt
    
    # Normalize harmonic amplitudes
    harmonic_sum = harmonic_amplitudes.sum(dim=1, keepdim=True).clamp(min=1e-8)
    harmonic_amplitudes = harmonic_amplitudes / harmonic_sum
    
    # Create noise magnitudes - higher during consonants, lower during vowels
    noise_magnitudes = torch.zeros(batch_size, n_noise_bands, n_frames, device=device)
    
    # Basic spectral shape for noise (formant-influenced)
    for b in range(batch_size):
        # Create band frequencies (mel-spaced)
        band_freqs = torch.linspace(0, sample_rate//2, n_noise_bands, device=device)
        
        for t in range(n_frames):
            # More noise during note transitions (consonants)
            is_transition = False
            for i in range(1, len(note_pattern)):
                transition_start = i * note_duration - 10
                transition_end = i * note_duration + 10
                if 0 <= transition_start < n_frames and 0 <= transition_end < n_frames:
                    if transition_start <= t < transition_end:
                        is_transition = True
                        break
            
            # Base spectral shape (formant-influenced)
            formant1_noise = torch.exp(-((band_freqs - formant1_freq[t])**2) / (2 * (formant1_bw*3)**2))
            formant2_noise = torch.exp(-((band_freqs - formant2_freq[t])**2) / (2 * (formant2_bw*3)**2))
            spectral_shape = formant1_noise + 0.5 * formant2_noise
            
            # More noise during transitions, less during stable vowels
            noise_level = 0.7 if is_transition else 0.2
            
            # Zero noise for unvoiced frames
            if f0[b, t] <= 0:
                noise_level = 0.0
                
            noise_magnitudes[b, :, t] = spectral_shape * noise_level
    
    # Generate transient events (consonants)
    transient_timings = torch.zeros(batch_size, max_transients, device=device)
    transient_ids = torch.zeros(batch_size, max_transients, device=device).long()
    transient_gains = torch.zeros(batch_size, max_transients, device=device)
    
    for b in range(batch_size):
        # Place transients at note transitions
        count = 0
        for i in range(1, len(note_pattern)):
            if count >= max_transients:
                break
                
            # Time at note transition
            transition_time = i * note_duration / 100.0  # Convert to seconds
            
            # Skip if beyond audio duration
            if transition_time >= duration_seconds:
                continue
                
            # Add transient
            transient_timings[b, count] = transition_time
            transient_ids[b, count] = torch.randint(0, n_transients, (1,), device=device)
            transient_gains[b, count] = 0.8 + 0.2 * torch.rand(1, device=device)
            count += 1
    
    # Create component gains (mix between harmonic, noise, transient)
    component_gains = torch.zeros(batch_size, 3, n_frames, device=device)
    
    for b in range(batch_size):
        # Default mix for sustained vowels: 80% harmonic, 20% noise
        component_gains[b, 0] = 0.8  # harmonic
        component_gains[b, 1] = 0.2  # noise
        component_gains[b, 2] = 0.0  # transient (applied separately)
        
        # Different mix for transitions: 50% harmonic, 30% noise, 20% transient
        for i in range(1, len(note_pattern)):
            transition_start = i * note_duration - 10
            transition_end = i * note_duration + 10
            
            if 0 <= transition_start < n_frames and 0 <= transition_end < n_frames:
                component_gains[b, 0, transition_start:transition_end] = 0.5  # Less harmonic
                component_gains[b, 1, transition_start:transition_end] = 0.3  # More noise
                component_gains[b, 2, transition_start:transition_end] = 0.2  # Add transients
    
    return {
        'f0': f0,
        'harmonic_amplitudes': harmonic_amplitudes,
        'noise_magnitudes': noise_magnitudes,
        'transient_timings': transient_timings,
        'transient_ids': transient_ids,
        'transient_gains': transient_gains,
        'component_gains': component_gains
    }
    
    
def plot_synthesized_audio(audio, f0, harmonic_amps, noise_mags, component_gains, sample_rate=16000):
    """
    Plot the synthesized audio and its components.
    """
    # Convert to numpy
    audio_np = audio.cpu().numpy()
    f0_np = f0.cpu().numpy()
    harmonic_amps_np = harmonic_amps.cpu().numpy()
    noise_mags_np = noise_mags.cpu().numpy()
    component_gains_np = component_gains.cpu().numpy()
    
    plt.figure(figsize=(12, 12))
    
    # Plot waveform
    plt.subplot(5, 1, 1)
    plt.plot(audio_np)
    plt.title('Generated Singing Voice Waveform')
    plt.xlabel('Samples')
    plt.ylabel('Amplitude')
    
    # Plot spectrogram
    plt.subplot(5, 1, 2)
    D = librosa.amplitude_to_db(np.abs(librosa.stft(audio_np)), ref=np.max)
    librosa.display.specshow(D, sr=sample_rate, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram')
    
    # Plot F0 contour
    plt.subplot(5, 1, 3)
    plt.plot(f0_np[0])
    plt.title('F0 Contour')
    plt.xlabel('Frames')
    plt.ylabel('Frequency (Hz)')
    
    # Plot harmonics (mean across time)
    plt.subplot(5, 1, 4)
    mean_harmonics = harmonic_amps_np[0].mean(axis=1)
    plt.bar(range(1, len(mean_harmonics) + 1), mean_harmonics)
    plt.title('Mean Harmonic Amplitudes')
    plt.xlabel('Harmonic Number')
    plt.ylabel('Amplitude')
    
    # Plot component gains over time
    plt.subplot(5, 1, 5)
    plt.plot(component_gains_np[0, 0], label='Harmonic')
    plt.plot(component_gains_np[0, 1], label='Noise')
    plt.plot(component_gains_np[0, 2], label='Transient')
    plt.title('Component Gains')
    plt.xlabel('Frames')
    plt.ylabel('Gain')
    plt.legend()
    
    plt.tight_layout()
    
    return plt
