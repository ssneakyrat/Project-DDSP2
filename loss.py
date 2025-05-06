import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio


class MultiResolutionSTFTLoss(nn.Module):
    """
    Multi-resolution STFT loss for comparing synthesized and ground truth audio.
    Computes loss in multiple time-frequency resolutions for better reconstruction.
    """
    def __init__(self, fft_sizes=[512, 1024, 2048], hop_sizes=[128, 256, 512], win_lengths=[512, 1024, 2048]):
        super(MultiResolutionSTFTLoss, self).__init__()
        
        assert len(fft_sizes) == len(hop_sizes) == len(win_lengths)
        
        self.fft_sizes = fft_sizes
        self.hop_sizes = hop_sizes
        self.win_lengths = win_lengths
        
    def stft_loss(self, x, y, fft_size, hop_size, win_length):
        """
        Calculate STFT loss for a specific FFT configuration.
        """
        # STFT transformations
        x_stft = torch.stft(x, fft_size, hop_size, win_length, return_complex=True)
        y_stft = torch.stft(y, fft_size, hop_size, win_length, return_complex=True)
        
        # Calculate magnitudes
        x_mag = torch.abs(x_stft)
        y_mag = torch.abs(y_stft)
        
        # Spectral convergence loss
        sc_loss = torch.norm(y_mag - x_mag, p='fro') / (torch.norm(y_mag, p='fro') + 1e-7)
        
        # Log-magnitude loss
        log_mag_loss = F.l1_loss(torch.log(x_mag + 1e-7), torch.log(y_mag + 1e-7))
        
        # Total loss
        total_loss = sc_loss + log_mag_loss
        
        return total_loss, sc_loss, log_mag_loss
    
    def forward(self, pred_audio, target_audio):
        """
        Calculate multi-resolution STFT loss.
        
        Args:
            pred_audio: Predicted waveform [batch_size, samples]
            target_audio: Target waveform [batch_size, samples]
            
        Returns:
            total_loss, spectral_convergence_loss, log_magnitude_loss
        """
        total_loss = 0.0
        sc_loss = 0.0
        log_mag_loss = 0.0
        
        # Apply STFT loss at each resolution
        for fft_size, hop_size, win_length in zip(self.fft_sizes, self.hop_sizes, self.win_lengths):
            res_loss, res_sc_loss, res_log_mag_loss = self.stft_loss(
                pred_audio, target_audio, fft_size, hop_size, win_length
            )
            
            total_loss += res_loss
            sc_loss += res_sc_loss
            log_mag_loss += res_log_mag_loss
            
        # Average across resolutions
        n_res = len(self.fft_sizes)
        return total_loss / n_res, sc_loss / n_res, log_mag_loss / n_res


class MelSpectrogramLoss(nn.Module):
    """
    Mel-spectrogram loss for comparing synthesized and ground truth audio.
    Useful for capturing perceptual differences in the frequency domain.
    """
    def __init__(self, sample_rate=16000, n_fft=1024, hop_length=256, n_mels=80, f_min=0, f_max=8000):
        super(MelSpectrogramLoss, self).__init__()
        
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            f_min=f_min,
            f_max=f_max,
            n_mels=n_mels,
            center=True,
            power=2.0,
        )
        
        self.log_transform = torchaudio.transforms.AmplitudeToDB()
    
    def forward(self, pred_audio, target_audio):
        """
        Calculate mel-spectrogram loss.
        
        Args:
            pred_audio: Predicted waveform [batch_size, samples]
            target_audio: Target waveform [batch_size, samples]
            
        Returns:
            mel_l1_loss, mel_l2_loss
        """
        # Calculate mel spectrograms
        pred_mel = self.mel_transform(pred_audio)
        target_mel = self.mel_transform(target_audio)
        
        # Convert to log scale
        pred_log_mel = self.log_transform(pred_mel)
        target_log_mel = self.log_transform(target_mel)
        
        # Calculate L1 and L2 losses
        mel_l1_loss = F.l1_loss(pred_log_mel, target_log_mel)
        mel_l2_loss = F.mse_loss(pred_log_mel, target_log_mel)
        
        return mel_l1_loss, mel_l2_loss


class DDSPSVSLoss(nn.Module):
    """
    Combined loss function for the DDSP-SVS model.
    Combines mel-spectrogram and multi-resolution STFT losses.
    """
    def __init__(self, sample_rate=16000, use_mel_loss=True, use_stft_loss=True):
        super(DDSPSVSLoss, self).__init__()
        
        self.use_mel_loss = use_mel_loss
        self.use_stft_loss = use_stft_loss
        
        if use_mel_loss:
            self.mel_loss = MelSpectrogramLoss(sample_rate=sample_rate)
        
        if use_stft_loss:
            self.stft_loss = MultiResolutionSTFTLoss()
            
        # Weights for different loss components
        self.mel_weight = 10.0
        self.stft_weight = 1.0
        
    def forward(self, model_outputs, targets):
        """
        Calculate the combined loss.
        
        Args:
            model_outputs: dict containing:
                audio: Synthesized audio [batch_size, samples]
                harmonic_amplitudes: [batch_size, n_harmonics, seq_length]
                noise_magnitudes: [batch_size, n_noise_bands, seq_length]
                transient_params: dict with timings, ids, gains
                component_gains: [batch_size, 3, seq_length]
                
            targets: dict containing:
                audio: Target audio [batch_size, samples]
                
        Returns:
            total_loss, losses_dict
        """
        pred_audio = model_outputs['audio']
        target_audio = targets['audio']
        
        losses = {}
        total_loss = 0.0
        
        # Mel-spectrogram loss
        if self.use_mel_loss:
            mel_l1, mel_l2 = self.mel_loss(pred_audio, target_audio)
            losses['mel_l1'] = mel_l1
            losses['mel_l2'] = mel_l2
            total_loss += self.mel_weight * (mel_l1 + mel_l2)
            
        # Multi-resolution STFT loss
        if self.use_stft_loss:
            stft_total, stft_sc, stft_mag = self.stft_loss(pred_audio, target_audio)
            losses['stft_total'] = stft_total
            losses['stft_sc'] = stft_sc
            losses['stft_mag'] = stft_mag
            total_loss += self.stft_weight * stft_total
            
        # Add individual loss components to total
        losses['total'] = total_loss
        
        return total_loss, losses


class MelReconstructionLoss(nn.Module):
    """
    Loss function for the first stage of training where we only predict mel spectrograms.
    This is used before incorporating the DDSP synthesizer in the full end-to-end model.
    """
    def __init__(self):
        super(MelReconstructionLoss, self).__init__()
    
    def forward(self, pred_mel, target_mel):
        """
        Calculate the L1 and L2 losses between predicted and target mel spectrograms.
        
        Args:
            pred_mel: Predicted mel-spectrogram [batch_size, n_mels, time]
            target_mel: Target mel-spectrogram [batch_size, n_mels, time]
            
        Returns:
            total_loss, losses_dict
        """
        # L1 loss
        l1_loss = F.l1_loss(pred_mel, target_mel)
        
        # L2 loss
        l2_loss = F.mse_loss(pred_mel, target_mel)
        
        # Total loss
        total_loss = l1_loss + l2_loss
        
        return total_loss, {
            'mel_l1': l1_loss,
            'mel_l2': l2_loss,
            'total': total_loss
        }
