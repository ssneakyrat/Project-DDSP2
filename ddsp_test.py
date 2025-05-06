import torch
import matplotlib.pyplot as plt
from scipy.io import wavfile
import numpy as np
import os
import time

# Import DDSP components
from harmonic_generator import HarmonicGenerator
from filtered_noise_generator import FilteredNoiseGenerator
from transient_generator import TransientGenerator
from ddsp_synthesizer import DDSPSynthesizer, generate_test_singing_parameters, plot_synthesized_audio


def test_individual_components():
    """
    Test each component (harmonic, noise, transient) individually
    """
    print("Testing individual DDSP components...")
    
    # Common parameters
    sample_rate = 16000
    duration_seconds = 2.0
    n_frames = int(duration_seconds * 100)  # 10ms frames
    batch_size = 1
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs("ddsp_test_output", exist_ok=True)
    
    #========================= Test Harmonic Generator =========================
    print("\nTesting Harmonic Generator...")
    start_time = time.time()
    
    # Parameters for harmonic generator
    n_harmonics = 40
    
    # Create a test f0 contour (a simple pitch glide with vibrato)
    base_f0 = torch.linspace(220, 440, n_frames, device=device)  # A3 to A4 glide
    
    # Add vibrato (5 Hz)
    vibrato_rate = 5.0
    vibrato_depth = 20.0  # Hz
    vibrato = vibrato_depth * torch.sin(2 * np.pi * vibrato_rate * torch.linspace(0, duration_seconds, n_frames, device=device))
    f0 = base_f0 + vibrato
    f0 = f0.unsqueeze(0)  # Add batch dimension
    
    # Create harmonic amplitudes
    time_indexes = torch.linspace(0, 1, n_frames, device=device)
    harmonic_indexes = torch.arange(1, n_harmonics + 1, device=device).float()
    
    # Create two formant peaks that move over time
    formant1_freq = torch.linspace(600, 800, n_frames, device=device)  # Hz
    formant2_freq = torch.linspace(1200, 900, n_frames, device=device)  # Hz
    formant1_bw = 100.0  # Bandwidth in Hz
    formant2_bw = 120.0  # Bandwidth in Hz
    
    harmonic_amplitudes = torch.zeros(batch_size, n_harmonics, n_frames, device=device)
    
    for t in range(n_frames):
        f0_at_t = f0[0, t].item()
        harmonic_freqs = harmonic_indexes * f0_at_t
        
        # Formant filter responses
        formant1_response = torch.exp(-((harmonic_freqs - formant1_freq[t])**2) / (2 * formant1_bw**2))
        formant2_response = torch.exp(-((harmonic_freqs - formant2_freq[t])**2) / (2 * formant2_bw**2))
        
        # Combine formants and add spectral tilt (1/h falloff)
        spectral_tilt = 1.0 / harmonic_indexes
        total_response = formant1_response + 0.7 * formant2_response
        harmonic_amplitudes[0, :, t] = total_response * spectral_tilt
    
    # Instantiate model
    harmonic_gen = HarmonicGenerator(sample_rate=sample_rate, n_harmonics=n_harmonics).to(device)
    
    # Generate audio
    with torch.no_grad():
        harmonic_audio = harmonic_gen(f0, harmonic_amplitudes)
    
    harmonic_time = time.time() - start_time
    print(f"Harmonic synthesis completed in {harmonic_time:.2f} seconds")
    
    # Normalize audio
    harmonic_audio = harmonic_audio / harmonic_audio.abs().max()
    
    # Convert to numpy and save
    harmonic_audio_np = harmonic_audio[0].cpu().numpy()
    wavfile.write('ddsp_test_output/harmonic_component.wav', sample_rate, harmonic_audio_np)
    
    #========================= Test Noise Generator =========================
    print("\nTesting Filtered Noise Generator...")
    start_time = time.time()
    
    # Parameters for noise generator
    n_bands = 80
    
    # Create noise magnitudes with a formant-like structure
    noise_magnitudes = torch.zeros(batch_size, n_bands, n_frames, device=device)
    
    # Create band frequencies (mel-spaced)
    band_freqs = torch.linspace(0, sample_rate//2, n_bands, device=device)
    
    for t in range(n_frames):
        # Formant-based spectral shape for the noise
        formant1_noise = torch.exp(-((band_freqs - formant1_freq[t])**2) / (2 * (formant1_bw*3)**2))
        formant2_noise = torch.exp(-((band_freqs - formant2_freq[t])**2) / (2 * (formant2_bw*3)**2))
        
        # Combine formants with a temporal envelope
        temporal_env = 0.5 + 0.5 * torch.sin(2 * np.pi * 0.5 * time_indexes[t])
        noise_magnitudes[0, :, t] = (formant1_noise + 0.5 * formant2_noise) * temporal_env
    
    # Instantiate model
    noise_gen = FilteredNoiseGenerator(sample_rate=sample_rate, n_bands=n_bands).to(device)
    
    # Generate audio
    with torch.no_grad():
        # Use the simpler approach for testing
        noise_audio = noise_gen.forward_simple(noise_magnitudes)
    
    noise_time = time.time() - start_time
    print(f"Noise synthesis completed in {noise_time:.2f} seconds")
    
    # Normalize audio
    noise_audio = noise_audio / noise_audio.abs().max()
    
    # Convert to numpy and save
    noise_audio_np = noise_audio[0].cpu().numpy()
    wavfile.write('ddsp_test_output/noise_component.wav', sample_rate, noise_audio_np)
    
    #========================= Test Transient Generator =========================
    print("\nTesting Transient Generator...")
    start_time = time.time()
    
    # Parameters for transient generator
    n_transients = 20
    max_transients = 8
    
    # Instantiate model
    transient_gen = TransientGenerator(sample_rate=sample_rate, n_transients=n_transients).to(device)
    
    # Generate random transient events
    transient_timings, transient_ids, transient_gains = transient_gen.generate_random_transients(
        batch_size, n_transients_per_batch=max_transients, max_time=duration_seconds
    )
    
    # Generate audio
    with torch.no_grad():
        transient_audio = transient_gen(transient_timings, transient_ids, transient_gains)
    
    transient_time = time.time() - start_time
    print(f"Transient synthesis completed in {transient_time:.2f} seconds")
    
    # Normalize audio
    transient_audio = transient_audio / transient_audio.abs().max()
    
    # Convert to numpy and save
    transient_audio_np = transient_audio[0].cpu().numpy()
    wavfile.write('ddsp_test_output/transient_component.wav', sample_rate, transient_audio_np)
    
    print("\nIndividual component tests completed!")
    return harmonic_audio, noise_audio#, transient_audio


def test_combined_synthesizer():
    """
    Test the complete DDSP synthesizer with all components
    """
    print("\n========== Testing Complete DDSP Synthesizer ==========")
    start_time = time.time()
    
    # Parameters
    sample_rate = 16000
    duration_seconds = 3.0
    batch_size = 1
    n_harmonics = 40
    n_noise_bands = 80
    n_transients = 20
    max_transients = 8
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create the synthesizer
    synthesizer = DDSPSynthesizer(
        sample_rate=sample_rate,
        n_harmonics=n_harmonics,
        n_noise_bands=n_noise_bands,
        n_transients=n_transients
    ).to(device)
    
    # Generate test parameters for singing voice
    print("Generating test singing parameters...")
    params = generate_test_singing_parameters(
        batch_size=batch_size,
        duration_seconds=duration_seconds,
        sample_rate=sample_rate,
        n_harmonics=n_harmonics,
        n_noise_bands=n_noise_bands,
        n_transients=n_transients,
        max_transients=max_transients,
        device=device
    )
    
    # Generate audio
    print("Synthesizing singing voice...")
    with torch.no_grad():
        audio = synthesizer(
            f0=params['f0'],
            harmonic_amplitudes=params['harmonic_amplitudes'],
            noise_magnitudes=params['noise_magnitudes'],
            transient_timings=params['transient_timings'],
            transient_ids=params['transient_ids'],
            transient_gains=params['transient_gains'],
            component_gains=params['component_gains']
        )
    
    total_time = time.time() - start_time
    print(f"Complete synthesis completed in {total_time:.2f} seconds")
    
    # Create output directory
    os.makedirs("ddsp_test_output", exist_ok=True)
    
    # Save audio
    audio_np = audio[0].cpu().numpy()
    wavfile.write('ddsp_test_output/singing_voice_synthesis.wav', sample_rate, audio_np)
    
    # Plot results
    print("Generating plots...")
    plt = plot_synthesized_audio(
        audio=audio[0],
        f0=params['f0'],
        harmonic_amps=params['harmonic_amplitudes'],
        noise_mags=params['noise_magnitudes'],
        component_gains=params['component_gains'],
        sample_rate=sample_rate
    )
    plt.savefig('ddsp_test_output/singing_voice_analysis.png')
    plt.close()
    
    print("\nDDSP singing synthesis test completed!")
    print(f"Output files saved in 'ddsp_test_output' directory")
    
    return audio, params


if __name__ == "__main__":
    print("==================================================")
    print("DDSP Singing Voice Synthesis - Test Suite")
    print("==================================================")
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Test individual components
    components_audio = test_individual_components()
    
    # Test combined synthesizer
    singing_audio, params = test_combined_synthesizer()
    
    print("\nAll tests completed successfully!")
    print("Check the 'ddsp_test_output' directory for the generated audio and visualization.")
