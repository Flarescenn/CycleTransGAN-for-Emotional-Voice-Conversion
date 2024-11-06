import os
import numpy as np
import argparse
import time
import librosa
import soundfile as sf
from preprocess import *
from model import CycleGAN
import torch
import pyworld as pw
from torch.utils.data import Dataset, DataLoader
from datetime import datetime

class VoiceDataset(Dataset):
    def __init__(self, coded_sps_A_norm, coded_sps_B_norm, n_frames=128):
        self.coded_sps_A_norm = coded_sps_A_norm
        self.coded_sps_B_norm = coded_sps_B_norm
        self.n_frames = n_frames
        
    def __len__(self):
        return min(len(self.coded_sps_A_norm), len(self.coded_sps_B_norm))
    
    def __getitem__(self, idx):
        start_A = np.random.randint(0, self.coded_sps_A_norm[idx].shape[1] - self.n_frames + 1)
        start_B = np.random.randint(0, self.coded_sps_B_norm[idx].shape[1] - self.n_frames + 1)
        
        end_A = start_A + self.n_frames
        end_B = start_B + self.n_frames
        
        return (torch.FloatTensor(self.coded_sps_A_norm[idx][:, start_A:end_A]), 
                torch.FloatTensor(self.coded_sps_B_norm[idx][:, start_B:end_B]))

def preprocess(train_A_dir, train_B_dir, cache_dir, sampling_rate, frame_period, num_mcep):
    os.makedirs(cache_dir, exist_ok=True) #create if doesnt exist
    cache_file = os.path.join(cache_dir, 'preprocessed_data.npz')
    if os.path.exists(cache_file):
        print('Loading preprocessed data from cache...')
        cached_data = np.load(cache_file, allow_pickle=True)
        return (cached_data['coded_sps_A_norm'], cached_data['coded_sps_B_norm'],
                cached_data['log_f0s_mean_A'], cached_data['log_f0s_std_A'],
                cached_data['log_f0s_mean_B'], cached_data['log_f0s_std_B'],
                cached_data['coded_sps_A_mean'], cached_data['coded_sps_A_std'],
                cached_data['coded_sps_B_mean'], cached_data['coded_sps_B_std'])

    print('Preprocessing data...')
    start_time = time.time()
    # Load and preprocess training data
    wavs_A = load_wavs(wav_dir=train_A_dir, sr=sampling_rate)
    wavs_B = load_wavs(wav_dir=train_B_dir, sr=sampling_rate)

    f0s_A, timeaxes_A, sps_A, aps_A, coded_sps_A = world_encode_data(
        wavs=wavs_A, fs=sampling_rate, frame_period=frame_period, coded_dim=num_mcep)
    f0s_B, timeaxes_B, sps_B, aps_B, coded_sps_B = world_encode_data(
        wavs=wavs_B, fs=sampling_rate, frame_period=frame_period, coded_dim=num_mcep)

    log_f0s_mean_A, log_f0s_std_A = logf0_statistics(f0s_A)
    log_f0s_mean_B, log_f0s_std_B = logf0_statistics(f0s_B)

    print('Log Pitch A')
    print('Mean: %f, Std: %f' % (log_f0s_mean_A, log_f0s_std_A))
    print('Log Pitch B')
    print('Mean: %f, Std: %f' % (log_f0s_mean_B, log_f0s_std_B))     

    coded_sps_A_transposed = transpose_in_list(lst=coded_sps_A)
    coded_sps_B_transposed = transpose_in_list(lst=coded_sps_B)

    coded_sps_A_norm, coded_sps_A_mean, coded_sps_A_std = coded_sps_normalization_fit_transoform(
        coded_sps=coded_sps_A_transposed)
    coded_sps_B_norm, coded_sps_B_mean, coded_sps_B_std = coded_sps_normalization_fit_transoform(
        coded_sps=coded_sps_B_transposed)

    # Cache the preprocessed data
    np.savez(cache_file,
             coded_sps_A_norm=coded_sps_A_norm,
             coded_sps_B_norm=coded_sps_B_norm,
             log_f0s_mean_A=log_f0s_mean_A,
             log_f0s_std_A=log_f0s_std_A,
             log_f0s_mean_B=log_f0s_mean_B,
             log_f0s_std_B=log_f0s_std_B,
             coded_sps_A_mean=coded_sps_A_mean,
             coded_sps_A_std=coded_sps_A_std,
             coded_sps_B_mean=coded_sps_B_mean,
             coded_sps_B_std=coded_sps_B_std)

    end_time = time.time()
    time_elapsed = end_time - start_time
    print(f'Preprocessing Done. Time Elapsed: {int(time_elapsed) // 3600:02d}:{(int(time_elapsed) % 3600) // 60:02d}:{(int(time_elapsed) % 60):02d}')

    return (coded_sps_A_norm, coded_sps_B_norm,
            log_f0s_mean_A, log_f0s_std_A,
            log_f0s_mean_B, log_f0s_std_B,
            coded_sps_A_mean, coded_sps_A_std,
            coded_sps_B_mean, coded_sps_B_std)

def train(train_A_dir, train_B_dir, model_dir, model_name, random_seed, validation_A_dir, validation_B_dir, output_dir, tensorboard_log_dir, n_frames):
    
    # Set random seeds for reproducibility
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)

    # Hyperparameters
    if n_frames == 128:
        num_epochs = 250  # More epochs for foundational training
    elif n_frames == 256:
        num_epochs = 150  # Medium context refinement
    elif n_frames == 380:
        num_epochs = 80   # Final refinement
    else:
        num_epochs = 500  # Default case

    mini_batch_size = 1  
    initial_generator_lr = 0.0002
    initial_discriminator_lr = 0.0001
    sampling_rate = 24000
    num_mcep = 24
    frame_period = 5.0
    lambda_cycle = 10
    lambda_identity = 5

    
    cache_dir = os.path.join(model_dir, 'preprocessing_cache')
    os.makedirs(cache_dir, exist_ok=True)
     # Preprocess data or load from cache
    (coded_sps_A_norm, coded_sps_B_norm,
     log_f0s_mean_A, log_f0s_std_A,
     log_f0s_mean_B, log_f0s_std_B,
     coded_sps_A_mean, coded_sps_A_std,
     coded_sps_B_mean, coded_sps_B_std) = preprocess(train_A_dir, train_B_dir, cache_dir, sampling_rate, frame_period, num_mcep)

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    np.savez(os.path.join(model_dir, 'logf0s_normalization.npz'), mean_A=log_f0s_mean_A, std_A=log_f0s_std_A, mean_B=log_f0s_mean_B, std_B=log_f0s_std_B)
    np.savez(os.path.join(model_dir, 'mcep_normalization.npz'), mean_A=coded_sps_A_mean, std_A=coded_sps_A_std, mean_B=coded_sps_B_mean, std_B=coded_sps_B_std)

    if validation_A_dir is not None:
        validation_A_output_dir = os.path.join(output_dir, f'converted_A{n_frames}')
        os.makedirs(validation_A_output_dir, exist_ok=True)

    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Initialize CycleGAN model
    model = CycleGAN(
        num_features=num_mcep, 
        log_dir=tensorboard_log_dir + str(n_frames),
        initial_generator_lr=initial_generator_lr,
        initial_discriminator_lr=initial_discriminator_lr
    ).to(device)
    #model = CycleGAN(num_features=num_mcep, log_dir=tensorboard_log_dir + str(n_frames)).to(device)
    if n_frames != 128:
        model.load(os.path.join(model_dir, model_name))
    # Create dataset and dataloader
    dataset = VoiceDataset(coded_sps_A_norm, coded_sps_B_norm, n_frames=n_frames)
    dataloader = DataLoader(dataset, batch_size=mini_batch_size, shuffle=True, num_workers=2)

    for epoch in range(num_epochs):
        print(f'Epoch: {epoch}')

        start_time_epoch = time.time()
        running_generator_loss = 0
        running_discriminator_loss = 0

        for i, (real_A, real_B) in enumerate(dataloader):
            num_iterations = len(dataloader) * epoch + i

           # Adjust hyperparameters based on iterations
            current_lambda_identity = lambda_identity
            if num_iterations > 100000:
                current_lambda_identity = 0.5

            generator_loss, discriminator_loss = model.train_step(
                real_A, real_B,
                lambda_cycle=lambda_cycle,
                lambda_identity=current_lambda_identity
            )
            running_generator_loss += generator_loss
            running_discriminator_loss += discriminator_loss

            if i % 200 == 0:
                current_gen_lr = model.generator_optimizer.param_groups[0]['lr']
                current_disc_lr = model.discriminator_optimizer.param_groups[0]['lr']
                print('Iteration: {:07d}, Generator Learning Rate: {:.7f}, Discriminator Learning Rate: {:.7f}, '
                      'Generator Loss : {:.6f}, Discriminator Loss : {:.6f}'.format(
                    num_iterations, current_gen_lr, current_disc_lr, generator_loss, discriminator_loss))
                
        # Calculate epoch averages
        avg_generator_loss = running_generator_loss / len(dataloader)
        avg_discriminator_loss = running_discriminator_loss / len(dataloader)
        print(f'Epoch {epoch} Average Losses - Generator: {avg_generator_loss:.4f}, Discriminator: {avg_discriminator_loss:.4f}')


        model.save(directory=model_dir, filename=f'{n_frames}_{model_name}')

        end_time_epoch = time.time()
        time_elapsed_epoch = end_time_epoch - start_time_epoch
        print(f'Time Elapsed for This Epoch: {int(time_elapsed_epoch) // 3600:02d}:{(int(time_elapsed_epoch) % 3600) // 60:02d}:{(int(time_elapsed_epoch) % 60):02d}')

        # Generate validation data every 10 epochs
        if validation_A_dir is not None and epoch % 10 == 0:
            print('Generating Validation Data B from A...')
            for file in os.listdir(validation_A_dir):
                filepath = os.path.join(validation_A_dir, file)
                wav, _ = librosa.load(filepath, sr=sampling_rate, mono=True)
                wav = wav_padding(wav=wav, sr=sampling_rate, frame_period=frame_period, multiple=4)
                f0, timeaxis, sp, ap = world_decompose(wav=wav, fs=sampling_rate, frame_period=frame_period)
                f0_converted = pitch_conversion(f0=f0, mean_log_src=log_f0s_mean_A, std_log_src=log_f0s_std_A, mean_log_target=log_f0s_mean_B, std_log_target=log_f0s_std_B)
                coded_sp = world_encode_spectral_envelop(sp=sp, fs=sampling_rate, dim=num_mcep)
                coded_sp_transposed = coded_sp.T
                coded_sp_norm = (coded_sp_transposed - coded_sps_A_mean) / coded_sps_A_std
                coded_sp_converted_norm = model.test(inputs=torch.tensor([coded_sp_norm], dtype=torch.float32).to(device), direction='A2B').cpu().numpy()[0]
                coded_sp_converted = coded_sp_converted_norm * coded_sps_B_std + coded_sps_B_mean
                coded_sp_converted = coded_sp_converted.T
                coded_sp_converted = np.ascontiguousarray(coded_sp_converted)
                decoded_sp_converted = world_decode_spectral_envelop(coded_sp=coded_sp_converted, fs=sampling_rate)
                wav_transformed = world_speech_synthesis(f0=f0_converted, decoded_sp=decoded_sp_converted, ap=ap, fs=sampling_rate, frame_period=frame_period)
                wav_transformed = np.nan_to_num(wav_transformed)
                sf.write(os.path.join(validation_A_output_dir, f'{epoch}_{os.path.basename(file)}'), wav_transformed, sampling_rate)

    model.save(directory=model_dir, filename=model_name)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train CycleGAN model for datasets.')

    train_A_dir_default = './data/training/NEUTRAL'
    train_B_dir_default = './data/training/SURPRISE'
    model_dir_default = './model/neutral_to_surprise_mceps'
    model_name_default = 'neutral_to_surprise_mceps.ckpt'
    random_seed_default = 0
    validation_A_dir_default = './data/evaluation_all/NEUTRAL'
    validation_B_dir_default = './data/evaluation_all/SURPRISE'
    output_dir_default = './validation_output'
    tensorboard_log_dir_default = './log'

    parser.add_argument('--num_f', type=int, help='Frame length.')
    parser.add_argument('--train_A_dir', type=str, help='Directory for A.', default=train_A_dir_default)
    parser.add_argument('--train_B_dir', type=str, help='Directory for B.', default=train_B_dir_default)
    parser.add_argument('--model_dir', type=str, help='Directory for saving models.', default=model_dir_default)
    parser.add_argument('--model_name', type=str, help='File name for saving model.', default=model_name_default)
    parser.add_argument('--random_seed', type=int, help='Random seed for model training.', default=random_seed_default)
    parser.add_argument('--validation_A_dir', type=str, help='Validation A directory.', default=validation_A_dir_default)
    parser.add_argument('--validation_B_dir', type=str, help='Validation B directory.', default=validation_B_dir_default)
    parser.add_argument('--output_dir', type=str, help='Output directory for converted validation voices.', default=output_dir_default)
    parser.add_argument('--tensorboard_log_dir', type=str, help='TensorBoard log directory.', default=tensorboard_log_dir_default)

    argv = parser.parse_args()

    num_f = argv.num_f
    train_A_dir = argv.train_A_dir
    train_B_dir = argv.train_B_dir
    model_dir = argv.model_dir
    model_name = argv.model_name
    random_seed = argv.random_seed
    validation_A_dir = None if argv.validation_A_dir.lower() == 'none' else argv.validation_A_dir
    validation_B_dir = None if argv.validation_B_dir.lower() == 'none' else argv.validation_B_dir
    output_dir = argv.output_dir
    tensorboard_log_dir = argv.tensorboard_log_dir

    train(train_A_dir=train_A_dir, train_B_dir=train_B_dir, model_dir=model_dir, model_name=model_name, random_seed=random_seed, validation_A_dir=validation_A_dir, validation_B_dir=validation_B_dir, output_dir=output_dir, tensorboard_log_dir=tensorboard_log_dir, n_frames=num_f)
