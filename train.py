import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import time
import logging
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from scipy.io import wavfile

# Import custom modules
from model import DDSPSVS
from loss import DDSPSVSLoss, MelReconstructionLoss
from dataset import get_dataloader
from ddsp_synthesizer import DDSPSynthesizer, plot_synthesized_audio

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("DDSP-SVS Training")

def parse_args():
    parser = argparse.ArgumentParser(description="Train DDSP-SVS model")
    
    # Data parameters
    parser.add_argument("--data_dir", type=str, default="./datasets", help="Path to dataset directory")
    parser.add_argument("--cache_dir", type=str, default="./cache", help="Path to cache directory")
    
    # Model parameters
    parser.add_argument("--hidden_dim", type=int, default=512, help="Hidden dimension size")
    parser.add_argument("--n_harmonics", type=int, default=100, help="Number of harmonics")
    parser.add_argument("--n_noise_bands", type=int, default=80, help="Number of noise bands")
    parser.add_argument("--n_transients", type=int, default=20, help="Number of transient templates")
    parser.add_argument("--max_transients", type=int, default=8, help="Maximum transients per utterance")
    parser.add_argument("--sample_rate", type=int, default=16000, help="Audio sample rate")
    
    # Training parameters
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=0.0001, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device")
    parser.add_argument("--log_interval", type=int, default=10, help="Logging interval (batches)")
    parser.add_argument("--save_interval", type=int, default=1, help="Saving interval (epochs)")
    parser.add_argument("--checkpoint", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--output_dir", type=str, default="./output", help="Output directory")
    
    # Training strategy parameters
    parser.add_argument("--two_stage", action="store_true", help="Use two-stage training (mel prediction then DDSP)")
    parser.add_argument("--mel_epochs", type=int, default=50, help="Number of epochs for mel prediction in two-stage training")
    
    args = parser.parse_args()
    return args

def save_checkpoint(model, optimizer, scheduler, epoch, save_path, args):
    """Save model checkpoint."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "epoch": epoch,
        "args": args.__dict__,
    }
    
    torch.save(checkpoint, save_path)
    logger.info(f"Checkpoint saved to {save_path}")

def load_checkpoint(model, optimizer, scheduler, checkpoint_path, device):
    """Load model checkpoint."""
    if not os.path.exists(checkpoint_path):
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        return model, optimizer, scheduler, 0
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    if scheduler is not None and "scheduler_state_dict" in checkpoint and checkpoint["scheduler_state_dict"] is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    
    epoch = checkpoint["epoch"]
    
    logger.info(f"Loaded checkpoint from {checkpoint_path} (epoch {epoch})")
    
    return model, optimizer, scheduler, epoch

def train_mel_prediction_epoch(model, dataloader, optimizer, loss_fn, device, epoch, args):
    """Train for one epoch (first stage: mel prediction only)."""
    model.train()
    
    total_loss = 0.0
    loss_components = {
        "mel_l1": 0.0,
        "mel_l2": 0.0,
    }
    
    start_time = time.time()
    
    for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Mel Stage - Epoch {epoch}")):
        # Move data to device
        phone_seq = batch["phone_seq_mel"].to(device)
        f0 = batch["f0"].to(device)
        singer_id = batch["singer_id"].squeeze(1).to(device)
        language_id = batch["language_id"].squeeze(1).to(device)
        mel = batch["mel"].to(device)
        
        # Reset gradients
        optimizer.zero_grad()
        
        # Forward pass through model
        outputs = model(phone_seq, f0, singer_id, language_id)
        
        # Calculate loss (using MelReconstructionLoss)
        # In the first stage, we just predict mel specs, not DDSP params
        pred_mel = outputs["mel"]  # Assuming model outputs mel prediction
        loss, losses_dict = loss_fn(pred_mel, mel)
        
        # Backward pass
        loss.backward()
        
        # Update parameters
        optimizer.step()
        
        # Accumulate loss statistics
        total_loss += loss.item()
        for k, v in losses_dict.items():
            if k in loss_components:
                loss_components[k] += v.item()
        
        # Logging
        if (batch_idx + 1) % args.log_interval == 0:
            logger.info(f"Mel Stage - Epoch {epoch} | Batch {batch_idx+1}/{len(dataloader)} | "
                      f"Loss: {loss.item():.4f} | "
                      f"Time: {time.time() - start_time:.2f}s")
            
            # Reset timer
            start_time = time.time()
    
    # Calculate average losses
    avg_loss = total_loss / len(dataloader)
    avg_components = {k: v / len(dataloader) for k, v in loss_components.items()}
    
    return avg_loss, avg_components

def train_ddsp_epoch(model, ddsp_synthesizer, dataloader, optimizer, loss_fn, device, epoch, args):
    """Train for one epoch (second stage: full DDSP end-to-end)."""
    model.train()
    
    total_loss = 0.0
    loss_components = {
        "mel_l1": 0.0,
        "mel_l2": 0.0,
        "stft_total": 0.0,
        "stft_sc": 0.0,
        "stft_mag": 0.0,
    }
    
    start_time = time.time()
    
    for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"DDSP Stage - Epoch {epoch}")):
        # Move data to device
        phone_seq = batch["phone_seq_mel"].to(device)
        f0 = batch["f0"].to(device)
        singer_id = batch["singer_id"].squeeze(1).to(device)
        language_id = batch["language_id"].squeeze(1).to(device)
        audio = batch["audio"].to(device)
        
        # Reset gradients
        optimizer.zero_grad()
        
        # Forward pass through model
        outputs = model(phone_seq, f0, singer_id, language_id)
        
        # Forward pass through DDSP synthesizer
        synthesized_audio = ddsp_synthesizer(
            f0=f0,
            harmonic_amplitudes=outputs["harmonic_amplitudes"],
            noise_magnitudes=outputs["noise_magnitudes"],
            transient_timings=outputs["transient_params"]["timings"],
            transient_ids=outputs["transient_params"]["ids"],
            transient_gains=outputs["transient_params"]["gains"],
            component_gains=outputs["component_gains"]
        )
        
        # Add synthesized audio to outputs
        outputs["audio"] = synthesized_audio
        
        # Calculate loss
        targets = {"audio": audio}
        loss, losses_dict = loss_fn(outputs, targets)
        
        # Backward pass
        loss.backward()
        
        # Update parameters
        optimizer.step()
        
        # Accumulate loss statistics
        total_loss += loss.item()
        for k, v in losses_dict.items():
            if k in loss_components and isinstance(v, torch.Tensor):
                loss_components[k] += v.item()
        
        # Logging
        if (batch_idx + 1) % args.log_interval == 0:
            logger.info(f"DDSP Stage - Epoch {epoch} | Batch {batch_idx+1}/{len(dataloader)} | "
                      f"Loss: {loss.item():.4f} | "
                      f"Time: {time.time() - start_time:.2f}s")
            
            # Reset timer
            start_time = time.time()
    
    # Calculate average losses
    avg_loss = total_loss / len(dataloader)
    avg_components = {k: v / len(dataloader) for k, v in loss_components.items()}
    
    return avg_loss, avg_components

def validate(model, ddsp_synthesizer, dataloader, loss_fn, device, epoch, args, stage="ddsp"):
    """Validate the model."""
    model.eval()
    
    total_loss = 0.0
    loss_components = {
        "mel_l1": 0.0,
        "mel_l2": 0.0,
        "stft_total": 0.0,
        "stft_sc": 0.0,
        "stft_mag": 0.0,
    }
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Validation - Epoch {epoch}")):
            # Move data to device
            phone_seq = batch["phone_seq_mel"].to(device)
            f0 = batch["f0"].to(device)
            singer_id = batch["singer_id"].squeeze(1).to(device)
            language_id = batch["language_id"].squeeze(1).to(device)
            audio = batch["audio"].to(device)
            mel = batch["mel"].to(device)
            
            # Forward pass through model
            outputs = model(phone_seq, f0, singer_id, language_id)
            
            if stage == "mel":
                # Use mel prediction loss
                pred_mel = outputs["mel"]
                loss, losses_dict = loss_fn(pred_mel, mel)
            else:
                # Forward pass through DDSP synthesizer
                synthesized_audio = ddsp_synthesizer(
                    f0=f0,
                    harmonic_amplitudes=outputs["harmonic_amplitudes"],
                    noise_magnitudes=outputs["noise_magnitudes"],
                    transient_timings=outputs["transient_params"]["timings"],
                    transient_ids=outputs["transient_params"]["ids"],
                    transient_gains=outputs["transient_params"]["gains"],
                    component_gains=outputs["component_gains"]
                )
                
                # Add synthesized audio to outputs
                outputs["audio"] = synthesized_audio
                
                # Calculate loss
                targets = {"audio": audio}
                loss, losses_dict = loss_fn(outputs, targets)
            
            # Accumulate loss statistics
            total_loss += loss.item()
            for k, v in losses_dict.items():
                if k in loss_components and isinstance(v, torch.Tensor):
                    loss_components[k] += v.item()
                    
            # Generate and save audio sample from the first batch
            if batch_idx == 0 and stage == "ddsp":
                # Save first example in batch for visualization
                example_idx = 0
                example_audio = synthesized_audio[example_idx].cpu().numpy()
                example_f0 = f0[example_idx].cpu()
                example_harm_amps = outputs["harmonic_amplitudes"][example_idx].cpu()
                example_noise_mags = outputs["noise_magnitudes"][example_idx].cpu()
                example_comp_gains = outputs["component_gains"][example_idx].cpu()
                
                # Create visualization
                plt_fig = plot_synthesized_audio(
                    example_audio, example_f0, example_harm_amps,
                    example_noise_mags, example_comp_gains,
                    sample_rate=args.sample_rate
                )
                
                # Save visualization
                os.makedirs(os.path.join(args.output_dir, "samples"), exist_ok=True)
                plt_fig.savefig(os.path.join(args.output_dir, "samples", f"val_sample_epoch_{epoch}.png"))
                plt.close(plt_fig)
                
                # Save audio sample
                sample_path = os.path.join(args.output_dir, "samples", f"val_sample_epoch_{epoch}.wav")
                wavfile.write(sample_path, args.sample_rate, (example_audio * 32767).astype(np.int16))
                
                # Also save the ground truth for comparison
                gt_audio = audio[example_idx].cpu().numpy()
                gt_path = os.path.join(args.output_dir, "samples", f"val_gt_epoch_{epoch}.wav")
                wavfile.write(gt_path, args.sample_rate, (gt_audio * 32767).astype(np.int16))
    
    # Calculate average losses
    avg_loss = total_loss / len(dataloader)
    avg_components = {k: v / len(dataloader) for k, v in loss_components.items()}
    
    return avg_loss, avg_components

def main():
    # Parse arguments
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set up TensorBoard writer
    writer = SummaryWriter(log_dir=os.path.join(args.output_dir, "logs"))
    
    # Set device
    device = torch.device(args.device)
    logger.info(f"Using device: {device}")
    
    # Load datasets
    logger.info("Loading datasets...")
    train_loader, val_loader, train_dataset, val_dataset = get_dataloader(
        batch_size=args.batch_size,
        num_workers=4,
        pin_memory=True,
        device=args.device
    )
    
    # Get vocabulary sizes
    n_phones = len(train_dataset.phone_map) + 1  # +1 for padding
    n_singers = len(train_dataset.singer_map)
    n_languages = len(train_dataset.language_map)
    
    logger.info(f"Dataset loaded with {len(train_dataset)} training and {len(val_dataset)} validation samples")
    logger.info(f"Vocabulary: {n_phones} phones, {n_singers} singers, {n_languages} languages")
    
    # Create model
    logger.info("Creating model...")
    model = DDSPSVS(
        n_phones=n_phones,
        n_singers=n_singers,
        n_languages=n_languages,
        hidden_dim=args.hidden_dim,
        n_harmonics=args.n_harmonics,
        n_noise_bands=args.n_noise_bands,
        n_transients=args.n_transients,
        max_transients=args.max_transients,
        sample_rate=args.sample_rate
    ).to(device)
    
    # Create DDSP synthesizer
    ddsp_synthesizer = DDSPSynthesizer(
        sample_rate=args.sample_rate,
        n_harmonics=args.n_harmonics,
        n_noise_bands=args.n_noise_bands,
        n_transients=args.n_transients
    ).to(device)
    
    # Create optimizers and loss functions
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, verbose=True
    )
    
    # Different loss functions for different stages
    mel_loss_fn = MelReconstructionLoss()
    ddsp_loss_fn = DDSPSVSLoss(sample_rate=args.sample_rate)
    
    # Load checkpoint if specified
    start_epoch = 0
    current_stage = "mel" if args.two_stage else "ddsp"
    
    if args.checkpoint:
        model, optimizer, scheduler, start_epoch = load_checkpoint(
            model, optimizer, scheduler, args.checkpoint, device
        )
        
        # Determine current stage based on checkpoint epoch
        if args.two_stage:
            current_stage = "mel" if start_epoch < args.mel_epochs else "ddsp"
    
    # Training loop
    logger.info(f"Starting training in {current_stage} stage...")
    best_val_loss = float("inf")
    
    for epoch in range(start_epoch, args.epochs):
        # Determine current training stage
        if args.two_stage and epoch == args.mel_epochs:
            logger.info("Switching from mel prediction to DDSP stage")
            current_stage = "ddsp"
        
        # Train for one epoch
        if current_stage == "mel":
            train_loss, train_components = train_mel_prediction_epoch(
                model, train_loader, optimizer, mel_loss_fn, device, epoch, args
            )
            val_loss, val_components = validate(
                model, ddsp_synthesizer, val_loader, mel_loss_fn, device, epoch, args, stage="mel"
            )
        else:  # DDSP stage
            train_loss, train_components = train_ddsp_epoch(
                model, ddsp_synthesizer, train_loader, optimizer, ddsp_loss_fn, device, epoch, args
            )
            val_loss, val_components = validate(
                model, ddsp_synthesizer, val_loader, ddsp_loss_fn, device, epoch, args, stage="ddsp"
            )
        
        # Update learning rate scheduler
        scheduler.step(val_loss)
        
        # Log metrics
        logger.info(f"Epoch {epoch} | Stage: {current_stage} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        # Write to TensorBoard
        writer.add_scalar(f"Loss/{current_stage}/train", train_loss, epoch)
        writer.add_scalar(f"Loss/{current_stage}/val", val_loss, epoch)
        
        for k, v in train_components.items():
            writer.add_scalar(f"Train/{current_stage}/{k}", v, epoch)
        
        for k, v in val_components.items():
            writer.add_scalar(f"Val/{current_stage}/{k}", v, epoch)
        
        # Save model if it's the best so far
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                model, optimizer, scheduler, epoch,
                os.path.join(args.output_dir, f"best_model_{current_stage}.pth"),
                args
            )
        
        # Save periodic checkpoint
        if (epoch + 1) % args.save_interval == 0:
            save_checkpoint(
                model, optimizer, scheduler, epoch,
                os.path.join(args.output_dir, f"checkpoint_epoch_{epoch}.pth"),
                args
            )
    
    # Save final model
    save_checkpoint(
        model, optimizer, scheduler, args.epochs - 1,
        os.path.join(args.output_dir, "final_model.pth"),
        args
    )
    
    logger.info("Training complete!")
    writer.close()

if __name__ == "__main__":
    main()
