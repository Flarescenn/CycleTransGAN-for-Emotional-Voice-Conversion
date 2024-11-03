import os
import numpy as np
import argparse
import time
import librosa
from preprocess import *
from model_f0 import CycleGAN  
from utils import *
from torch.utils.data import Dataset, DataLoader
import torch

class F0Dataset(Dataset):
    def __init__(self, lf0_cwt_norm, n_frames=512):
        self.lf0_cwt_norm = lf0_cwt_norm
        self.n_frames = n_frames
        
    def __len__(self):
        return len(self.lf0_cwt_norm)
    
    def __getitem__(self, idx):
        data = self.lf0_cwt_norm[idx]
        if data.shape[1] < self.n_frames:
            # Pad if shorter than n_frames
            pad_length = self.n_frames - data.shape[1]
            data = np.pad(data, ((0, 0), (0, pad_length)), 'constant')
        elif data.shape[1] > self.n_frames:
            # Randomly sample a segment of n_frames
            start = np.random.randint(0, data.shape[1] - self.n_frames + 1)
            data = data[:, start:start + self.n_frames]
        
        return torch.FloatTensor(data)
    
def train(train_A_dir, train_B_dir, model_dir, model_name, random_seed, validation_A_dir, validation_B_dir, output_dir, tensorboard_log_dir):
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)
    
    num_epochs = 6800
    mini_batch_size = 1
    generator_lr = 0.0001
    generator_lr_decay = generator_lr / 10000000
    discriminator_lr = 0.001
    discriminator_lr_decay = discriminator_lr / 10000000
    sampling_rate = 24000
    num_mcep = 24
    num_scale = 10
    frame_period = 5.0
    n_frames = 512
    lambda_cycle = 10
    lambda_identity = 5

    print('Preprocessing Data...')
    start_time = time.time()

    # Load and encode data
    wavs_A = load_wavs(wav_dir=train_A_dir, sr=sampling_rate)
    wavs_B = load_wavs(wav_dir=train_B_dir, sr=sampling_rate)

    f0s_A, timeaxes_A, sps_A, aps_A, coded_sps_A = world_encode_data(
        wavs=wavs_A, fs=sampling_rate, frame_period=frame_period, coded_dim=num_mcep)
    f0s_B, timeaxes_B, sps_B, aps_B, coded_sps_B = world_encode_data(
        wavs=wavs_B, fs=sampling_rate, frame_period=frame_period, coded_dim=num_mcep)

    # Log F0 statistics
    log_f0s_mean_A, log_f0s_std_A = logf0_statistics(f0s_A)
    log_f0s_mean_B, log_f0s_std_B = logf0_statistics(f0s_B)

    print('Log Pitch A')
    print('Mean: %f, Std: %f' % (log_f0s_mean_A, log_f0s_std_A))
    print('Log Pitch B')
    print('Mean: %f, Std: %f' % (log_f0s_mean_B, log_f0s_std_B))

    # Get CWT features
    lf0_cwt_norm_A, scales_A, means_A, stds_A = get_lf0_cwt_norm(
        f0s_A, mean=log_f0s_mean_A, std=log_f0s_std_A)
    lf0_cwt_norm_B, scales_B, means_B, stds_B = get_lf0_cwt_norm(
        f0s_B, mean=log_f0s_mean_B, std=log_f0s_std_B)

    # Transpose for processing
    lf0_cwt_norm_A_transposed = transpose_in_list(lst=lf0_cwt_norm_A)
    lf0_cwt_norm_B_transposed = transpose_in_list(lst=lf0_cwt_norm_B)

    # Save normalization parameters
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    np.savez(os.path.join(model_dir, 'logf0s_normalization.npz'),
             mean_A=log_f0s_mean_A, std_A=log_f0s_std_A,
             mean_B=log_f0s_mean_B, std_B=log_f0s_std_B)

# Create output directories
    if validation_A_dir is not None:
        validation_A_output_dir = os.path.join(output_dir, 'converted_A')
        os.makedirs(validation_A_output_dir, exist_ok=True)

    if validation_B_dir is not None:
        validation_B_output_dir = os.path.join(output_dir, 'converted_B')
        os.makedirs(validation_B_output_dir, exist_ok=True)

    # Create datasets and dataloaders
    dataset_A = F0Dataset(lf0_cwt_norm_A_transposed, n_frames=n_frames)
    dataset_B = F0Dataset(lf0_cwt_norm_B_transposed, n_frames=n_frames)
    
    dataloader_A = DataLoader(dataset_A, batch_size=mini_batch_size, shuffle=True)
    dataloader_B = DataLoader(dataset_B, batch_size=mini_batch_size, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CycleGAN(num_features=num_scale, log_dir=tensorboard_log_dir).to(device)

    end_time = time.time()
    time_elapsed = end_time - start_time
    print('Preprocessing Done.')
    print('Time Elapsed for Data Preprocessing: %02d:%02d:%02d' % 
          (time_elapsed // 3600, (time_elapsed % 3600 // 60), (time_elapsed % 60 // 1)))

    # Training loop
    for epoch in range(num_epochs):
        print('Epoch: %d' % epoch)
        start_time_epoch = time.time()

        # Create iterators for both dataloaders
        iterator_A = iter(dataloader_A)
        iterator_B = iter(dataloader_B)
        
        num_iterations = min(len(dataloader_A), len(dataloader_B))

        for i in range(num_iterations):
            try:
                input_A = next(iterator_A)
                input_B = next(iterator_B)
            except StopIteration:
                iterator_A = iter(dataloader_A)
                iterator_B = iter(dataloader_B)
                input_A = next(iterator_A)
                input_B = next(iterator_B)

            # Adjust learning rates and lambda_identity based on iterations
            current_iteration = num_iterations * epoch + i

            if current_iteration > 10000:
                lambda_identity = 0.5
            if current_iteration > 20000:
                generator_learning_rate = max(0.00001, 
                    generator_learning_rate - generator_lr_decay)
                discriminator_learning_rate = max(0.00001, 
                    discriminator_learning_rate - discriminator_lr_decay)

            # Train step
            generator_loss, discriminator_loss = model.train_step(
                input_A=input_A,
                input_B=input_B,
                lambda_cycle=lambda_cycle,
                lambda_identity=lambda_identity,
                generator_learning_rate=generator_learning_rate,
                discriminator_learning_rate=discriminator_learning_rate
            )

            if i % 50 == 0:
                print('Iteration: {:07d}, Generator Learning Rate: {:.7f}, '
                      'Discriminator Learning Rate: {:.7f}, Generator Loss : {:.3f}, '
                      'Discriminator Loss : {:.3f}'.format(
                    current_iteration, generator_learning_rate, 
                    discriminator_learning_rate, generator_loss, discriminator_loss))

        # Save model after each epoch
        model.save(directory=model_dir, filename=model_name)
        print(f'Saved model checkpoint at epoch {epoch + 1}')
        end_time_epoch = time.time()
        time_elapsed_epoch = end_time_epoch - start_time_epoch
        print('Time Elapsed for This Epoch: %02d:%02d:%02d' % 
              (time_elapsed_epoch // 3600, (time_elapsed_epoch % 3600 // 60), 
               (time_elapsed_epoch % 60 // 1)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train CycleGAN model for F0 conversion.')

    train_A_dir_default = './data/training/NEUTRAL'
    train_B_dir_default = './data/training/SURPRISE'
    model_dir_default = './model/neutral_surprise_f0'
    model_name_default = 'neutral_to_surprise_f0.ckpt'
    random_seed_default = 0
    validation_A_dir_default = './data/evaluation_all/NEUTRAL'
    validation_B_dir_default = './data/evaluation_all/SURPRISE'
    output_dir_default = './validation_output'
    tensorboard_log_dir_default = './log'

    parser.add_argument('--train_A_dir', type=str, help='Directory for A.',
                       default=train_A_dir_default)
    parser.add_argument('--train_B_dir', type=str, help='Directory for B.',
                       default=train_B_dir_default)
    parser.add_argument('--model_dir', type=str, help='Directory for saving models.',
                       default=model_dir_default)
    parser.add_argument('--model_name', type=str, help='File name for saving model.',
                       default=model_name_default)
    parser.add_argument('--random_seed', type=int, help='Random seed for training.',
                       default=random_seed_default)
    parser.add_argument('--validation_A_dir', type=str,
                       help='Validation A directory.',
                       default=validation_A_dir_default)
    parser.add_argument('--validation_B_dir', type=str,
                       help='Validation B directory.',
                       default=validation_B_dir_default)
    parser.add_argument('--output_dir', type=str,
                       help='Output directory for converted voices.',
                       default=output_dir_default)
    parser.add_argument('--tensorboard_log_dir', type=str,
                       help='TensorBoard log directory.',
                       default=tensorboard_log_dir_default)

    argv = parser.parse_args()

    train_A_dir = argv.train_A_dir
    train_B_dir = argv.train_B_dir
    model_dir = argv.model_dir
    model_name = argv.model_name
    random_seed = argv.random_seed
    validation_A_dir = None if argv.validation_A_dir.lower() == 'none' else argv.validation_A_dir
    validation_B_dir = None if argv.validation_B_dir.lower() == 'none' else argv.validation_B_dir
    output_dir = argv.output_dir
    tensorboard_log_dir = argv.tensorboard_log_dir

    train(train_A_dir=train_A_dir, train_B_dir=train_B_dir,
          model_dir=model_dir, model_name=model_name,
          random_seed=random_seed,
          validation_A_dir=validation_A_dir,
          validation_B_dir=validation_B_dir,
          output_dir=output_dir,
          tensorboard_log_dir=tensorboard_log_dir)
