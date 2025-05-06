"""
Configuration file for DDSP-SVS model hyperparameters.
"""

class ModelConfig:
    """Model architecture configuration"""
    # Core dimensions
    hidden_dim = 512
    
    # Encoder parameters
    text_encoder_layers = 3
    text_encoder_heads = 8
    text_encoder_dropout = 0.1
    
    f0_encoder_layers = 5
    f0_encoder_kernel = 3
    f0_encoder_dropout = 0.1
    
    # Decoder parameters
    decoder_layers = 2
    decoder_dropout = 0.1
    
    # DDSP parameters
    n_harmonics = 100
    n_noise_bands = 80
    n_transients = 20
    max_transients = 8
    
    # Template size for transients (in samples)
    transient_template_size = 1024


class TrainingConfig:
    """Training configuration"""
    # Basic training parameters
    batch_size = 16
    learning_rate = 0.0001
    max_epochs = 100
    
    # Two-stage training
    use_two_stage = True
    mel_epochs = 50
    
    # Optimizer parameters
    weight_decay = 0.0001
    grad_clip_thresh = 1.0
    
    # Scheduler parameters
    lr_decay_factor = 0.5
    lr_decay_patience = 5
    min_lr = 0.00001
    
    # Loss weights
    mel_weight = 10.0
    stft_weight = 1.0
    
    # Data parameters
    use_augmentation = True
    noise_scale = 0.01  # Scale of noise added during training
    
    # Dataset split
    train_files = None  # Use all available except validation
    val_files = 100     # Number of validation files
    
    # Seed for reproducibility
    seed = 42


class AudioConfig:
    """Audio processing configuration"""
    sample_rate = 16000
    n_mels = 80
    hop_length = 256
    win_length = 1024
    
    fmin = 40
    fmax = 8000
    
    # Context window in seconds for chunking audio
    context_window_sec = 2
    
    # STFT configuration for loss
    stft_sizes = [512, 1024, 2048]
    stft_hops = [128, 256, 512]
    stft_wins = [512, 1024, 2048]


class PathConfig:
    """File paths configuration"""
    dataset_dir = "./datasets"
    cache_dir = "./cache"
    output_dir = "./output"
    
    # Checkpoint paths
    checkpoint_path = None  # Path to resume training
    pretrained_path = None  # Path to pretrained model
    
    # Log directories
    log_dir = "./output/logs"
    sample_dir = "./output/samples"


def get_config():
    """Get all configurations in a single dictionary"""
    return {
        "model": ModelConfig(),
        "training": TrainingConfig(),
        "audio": AudioConfig(),
        "path": PathConfig()
    }
