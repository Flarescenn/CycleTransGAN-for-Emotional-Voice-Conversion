import os
import numpy as np
import argparse
import time
import librosa
import torch
from preprocess import *
from model_f0 import CycleGAN
from utils import get_lf0_cwt_norm, get_cont_lf0, get_lf0_cwt, inverse_cwt

def train(train_A_dir, train_B_dir, model_dir, model_name, random_seed, validation_A_dir, validation_B_dir, output_dir, tensorboard_log_dir):
    # Set random seed for reproducibility
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    # Hyperparameters
    num_epochs = 6800
    mini_batch_size = 1  # mini_batch_size = 1 is better for CycleGAN
    generator_learning_rate = 0.0001
    generator_learning_rate_decay = generator_learning_rate / 10000000
    discriminator_learning_rate = 0.001
    discriminator_learning_rate_decay = discriminator_learning_rate / 10000000
    sampling_rate = 24000
    num_mcep = 24
    num_scale = 10
    frame_period = 5.0
    n_frames = 512
    lambda_cycle = 10
    lambda_identity = 5

    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('Preprocessing Data...')
    start_time = time.time()

    # Load wav files and preprocess
    wavs_A = load_wavs(wav_dir=train_A_dir, sr=sampling_rate)
    wavs_B = load_wavs(wav_dir=train_B_dir, sr=sampling_rate)

    f0s_A, _, sps_A, aps_A, coded_sps_A = world_encode_data(wavs=wavs_A, fs=sampling_rate, frame_period=frame_period, coded_dim=num_mcep)
    f0s_B, _, sps_B, aps_B, coded_sps_B = world_encode_data(wavs=wavs_B, fs=sampling_rate, frame_period=frame_period, coded_dim=num_mcep)

    log_f0s_mean_A, log_f0s_std_A = logf0_statistics(f0s_A)
    log_f0s_mean_B, log_f0s_std_B = logf0_statistics(f0s_B)

    # Get continuous wavelet transform of lf0
    lf0_cwt_norm_A, scales_A, means_A, stds_A = get_lf0_cwt_norm(f0s_A, mean=log_f0s_mean_A, std=log_f0s_std_A)
    lf0_cwt_norm_B, scales_B, means_B, stds_B = get_lf0_cwt_norm(f0s_B, mean=log_f0s_mean_B, std=log_f0s_std_B)

    # Transpose the data for processing
    lf0_cwt_norm_A_transposed = transpose_in_list(lf0_cwt_norm_A)
    lf0_cwt_norm_B_transposed = transpose_in_list(lf0_cwt_norm_B)
    coded_sps_A_transposed = transpose_in_list(coded_sps_A)
    coded_sps_B_transposed = transpose_in_list(coded_sps_B)

    # Normalize the spectral data
    coded_sps_A_norm, coded_sps_A_mean, coded_sps_A_std = coded_sps_normalization_fit_transoform(coded_sps=coded_sps_A_transposed)
    coded_sps_B_norm, coded_sps_B_mean, coded_sps_B_std = coded_sps_normalization_fit_transoform(coded_sps=coded_sps_B_transposed)

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Save normalization data
    np.savez(os.path.join(model_dir, 'logf0s_normalization.npz'), mean_A=log_f0s_mean_A, std_A=log_f0s_std_A, mean_B=log_f0s_mean_B, std_B=log_f0s_std_B)
    np.savez(os.path.join(model_dir, 'mcep_normalization.npz'), mean_A=coded_sps_A_mean, std_A=coded_sps_A_std, mean_B=coded_sps_B_mean, std_B=coded_sps_B_std)

    if validation_A_dir is not None:
        validation_A_output_dir = os.path.join(output_dir, 'converted_A')
        os.makedirs(validation_A_output_dir, exist_ok=True)

    if validation_B_dir is not None:
        validation_B_output_dir = os.path.join(output_dir, 'converted_B')
        os.makedirs(validation_B_output_dir, exist_ok=True)

    end_time = time.time()
    time_elapsed = end_time - start_time
    print(f'Preprocessing Done. Time Elapsed: {time_elapsed // 3600:02d}:{(time_elapsed % 3600) // 60:02d}:{(time_elapsed % 60):02d}')

    # Initialize CycleGAN model
    num_feats = 10
    model = CycleGAN(num_features=num_feats, log_dir=tensorboard_log_dir).to(device)

    for epoch in range(num_epochs):
        print(f'Epoch: {epoch}')

        start_time_epoch = time.time()

        # Prepare training data
        data_As, data_Bs = [], []
        for lf0_a in lf0_cwt_norm_A_transposed:
            data_As.append(lf0_a)

        for lf0_b in lf0_cwt_norm_B_transposed:
            data_Bs.append(lf0_b)

        # Sample training data
        dataset_A, dataset_B = sample_train_data(dataset_A=data_As, dataset_B=data_Bs, n_frames=n_frames)
        n_samples = dataset_A.shape[0]

        for i in range(n_samples // mini_batch_size):
            num_iterations = n_samples // mini_batch_size * epoch + i

            if num_iterations > 10000:
                lambda_identity = 1
            if num_iterations > 20000:
                generator_learning_rate = max(0.00001, generator_learning_rate - generator_learning_rate_decay)
                discriminator_learning_rate = max(0.00001, discriminator_learning_rate - discriminator_learning_rate_decay)

            start, end = i * mini_batch_size, (i + 1) * mini_batch_size

            # Train model
            generator_loss, discriminator_loss = model.train(
                input_A=torch.tensor(dataset_A[start:end], dtype=torch.float32).to(device),
                input_B=torch.tensor(dataset_B[start:end], dtype=torch.float32).to(device),
                lambda_cycle=lambda_cycle,
                lambda_identity=lambda_identity,
                generator_lr=generator_learning_rate,
                discriminator_lr=discriminator_learning_rate
            )

            if i % 50 == 0:
                print(f'Iteration: {num_iterations:07d}, Generator LR: {generator_learning_rate:.7f}, Discriminator LR: {discriminator_learning_rate:.7f}, Generator Loss: {generator_loss:.3f}, Discriminator Loss: {discriminator_loss:.3f}')

        model.save(directory=model_dir, filename=model_name)

        end_time_epoch = time.time()
        time_elapsed_epoch = end_time_epoch - start_time_epoch
        print(f'Time Elapsed for This Epoch: {time_elapsed_epoch // 3600:02d}:{(time_elapsed_epoch % 3600) // 60:02d}:{(time_elapsed_epoch % 60):02d}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train CycleGAN model for datasets.')

    train_A_dir_default = './data/training/NEUTRAL'
    train_B_dir_default = './data/training/SURPRISE'
    model_dir_default = './model/neutral_surprise_f0'
    model_name_default = 'neutral_to_surprise_f0.ckpt'
    random_seed_default = 0
    validation_A_dir_default = './data/evaluation_all/NEUTRAL'
    validation_B_dir_default = './data/evaluation_all/SURPRISE'
    output_dir_default = './validation_output'
    tensorboard_log_dir_default = './log'

    parser.add_argument('--train_A_dir', type=str, help='Directory for A.', default=train_A_dir_default)
    parser.add_argument('--train_B_dir', type=str, help='Directory for B.', default=train_B_dir_default)
    parser.add_argument('--model_dir', type=str, help='Directory for saving models.', default=model_dir_default)
    parser.add_argument('--model_name', type=str, help='File name for saving model.', default=model_name_default)
    parser.add_argument('--random_seed', type=int, help='Random seed for model training.', default=random_seed_default)
    parser.add_argument('--validation_A_dir', type=str, help='Convert validation A after each training epoch.', default=validation_A_dir_default)
    parser.add_argument('--validation_B_dir', type=str, help='Convert validation B after each training epoch.', default=validation_B_dir_default)
    parser.add_argument('--output_dir', type=str, help='Output directory for converted validation voices.', default=output_dir_default)
    parser.add_argument('--tensorboard_log_dir', type=str, help='TensorBoard log directory.', default=tensorboard_log_dir_default)

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

    train(train_A_dir=train_A_dir, train_B_dir=train_B_dir, model_dir=model_dir, model_name=model_name, random_seed=random_seed, validation_A_dir=validation_A_dir, validation_B_dir=validation_B_dir, output_dir=output_dir, tensorboard_log_dir=tensorboard_log_dir)
