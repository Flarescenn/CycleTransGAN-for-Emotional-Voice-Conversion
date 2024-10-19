import os
import numpy as np
import librosa
import torch
import shutil
import soundfile as sf  
from train import train as train_vanilla
from train_f0 import train as train_f0

# Creating mock directories
def setup_mock_data():
    os.makedirs('./mock_data/training/NEUTRAL', exist_ok=True)
    os.makedirs('./mock_data/training/SURPRISE', exist_ok=True)
    os.makedirs('./mock_data/evaluation_all/NEUTRAL', exist_ok=True)
    os.makedirs('./mock_data/evaluation_all/SURPRISE', exist_ok=True)
    
    sr = 24000  # Sampling rate
    t = np.linspace(0, 5, sr * 5)  # 5 seconds of audio
    
    # Generate mock audio files
    for i in range(5):
        neutral_wav = np.sin(2 * np.pi * 440 * t).astype(np.float32)
        surprise_wav = np.sin(2 * np.pi * 880 * t).astype(np.float32)

        sf.write(f'./mock_data/training/NEUTRAL/sample_{i}.wav', neutral_wav, sr)
        sf.write(f'./mock_data/training/SURPRISE/sample_{i}.wav', surprise_wav, sr)

# Remove mock data
def cleanup():
    if os.path.exists('./mock_data'):
        shutil.rmtree('./mock_data')
    if os.path.exists('./mock_model'):
        shutil.rmtree('./mock_model')
    if os.path.exists('./mock_output'):
        shutil.rmtree('./mock_output')

# Testing vanilla CycleGAN training
def test_train_vanilla():
    train_A_dir = './mock_data/training/NEUTRAL'
    train_B_dir = './mock_data/training/SURPRISE'
    model_dir = './mock_model'
    output_dir = './mock_output'
    tensorboard_log_dir = './mock_logs'
    
    train_vanilla(
        train_A_dir=train_A_dir,
        train_B_dir=train_B_dir,
        model_dir=model_dir,
        model_name='mock_model.ckpt',
        random_seed=42,
        validation_A_dir=None,
        validation_B_dir=None,
        output_dir=output_dir,
        tensorboard_log_dir=tensorboard_log_dir,
        n_frames=128
    )

# Testing f0 CycleGAN training
def test_train_f0():
    train_A_dir = './mock_data/training/NEUTRAL'
    train_B_dir = './mock_data/training/SURPRISE'
    model_dir = './mock_model_f0'
    output_dir = './mock_output_f0'
    tensorboard_log_dir = './mock_logs_f0'
    
    train_f0(
        train_A_dir=train_A_dir,
        train_B_dir=train_B_dir,
        model_dir=model_dir,
        model_name='mock_model_f0.ckpt',
        random_seed=42,
        validation_A_dir=None,
        validation_B_dir=None,
        output_dir=output_dir,
        tensorboard_log_dir=tensorboard_log_dir
    )

if __name__ == '__main__':
    print("Setting up mock data...")
    setup_mock_data()

    print("Testing vanilla CycleGAN training...")
    test_train_vanilla()

    print("Testing f0 CycleGAN training...")
    test_train_f0()

    print("Cleaning up mock data...")
    cleanup()
