import os
import numpy as np
import argparse
import time
import librosa
from preprocess import *
from model_f0 import CycleGAN  
from utils import *

import torch

def train(train_A_dir, train_B_dir, model_dir, model_name, random_seed, validation_A_dir, validation_B_dir, output_dir, tensorboard_log_dir):
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    
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
    wavs_A = load_wavs(train_A_dir, sampling_rate)
    wavs_B = load_wavs(train_B_dir, sampling_rate)
    f0s_A, _, _, _, coded_sps_A = world_encode_data(wavs_A, sampling_rate, frame_period, num_mcep)
    f0s_B, _, _, _, coded_sps_B = world_encode_data(wavs_B, sampling_rate, frame_period, num_mcep)

    # Log F0 statistics
    log_f0s_mean_A, log_f0s_std_A = logf0_statistics(f0s_A)
    log_f0s_mean_B, log_f0s_std_B = logf0_statistics(f0s_B)

    # Get normalized continuous wavelet transform of lf0
    lf0_cwt_norm_A, _, _, _ = get_lf0_cwt_norm(f0s_A, log_f0s_mean_A, log_f0s_std_A)
    lf0_cwt_norm_B, _, _, _ = get_lf0_cwt_norm(f0s_B, log_f0s_mean_B, log_f0s_std_B)

    coded_sps_A_norm, coded_sps_A_mean, coded_sps_A_std = coded_sps_normalization_fit_transoform(transpose_in_list(coded_sps_A))
    coded_sps_B_norm, coded_sps_B_mean, coded_sps_B_std = coded_sps_normalization_fit_transoform(transpose_in_list(coded_sps_B))

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    np.savez(os.path.join(model_dir, 'logf0s_normalization.npz'), mean_A=log_f0s_mean_A, std_A=log_f0s_std_A, mean_B=log_f0s_mean_B, std_B=log_f0s_std_B)
    np.savez(os.path.join(model_dir, 'mcep_normalization.npz'), mean_A=coded_sps_A_mean, std_A=coded_sps_A_std, mean_B=coded_sps_B_mean, std_B=coded_sps_B_std)

    model = CycleGAN(num_features=num_scale).to('cuda' if torch.cuda.is_available() else 'cpu')

    optimizer_gen = torch.optim.Adam(model.generator.parameters(), lr=generator_lr)
    optimizer_disc = torch.optim.Adam(model.discriminator.parameters(), lr=discriminator_lr)
    criterion = torch.nn.MSELoss()

    for epoch in range(num_epochs):
        print(f'Epoch: {epoch + 1}/{num_epochs}')

        data_As = [lf0 for lf0 in transpose_in_list(lf0_cwt_norm_A)]
        data_Bs = [lf0 for lf0 in transpose_in_list(lf0_cwt_norm_B)]

        dataset_A, dataset_B = sample_train_data(data_As, data_Bs, n_frames)
        n_samples = dataset_A.shape[0]

        for i in range(n_samples // mini_batch_size):
            start = i * mini_batch_size
            end = (i + 1) * mini_batch_size
            input_A = torch.tensor(dataset_A[start:end]).float().to(model.device)
            input_B = torch.tensor(dataset_B[start:end]).float().to(model.device)

            optimizer_gen.zero_grad()
            optimizer_disc.zero_grad()

            gen_loss, disc_loss = model.train(input_A, input_B, lambda_cycle, lambda_identity)
            gen_loss.backward()
            disc_loss.backward()
            optimizer_gen.step()
            optimizer_disc.step()

            if i % 50 == 0:
                print(f'Iteration {i}, Gen Loss: {gen_loss.item()}, Disc Loss: {disc_loss.item()}')

        model.save(model_dir, model_name)
        print(f'Saved model checkpoint at epoch {epoch + 1}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train CycleGAN model with F0 features.')
    train_A_dir_default = './data/training/NEUTRAL'
    train_B_dir_default = './data/training/SURPRISE'
    model_dir_default = './model/neutral_to_surprise_mceps'
    model_name_default = 'neutral_to_surprise_mceps.ckpt'
    random_seed_default = 0
    validation_A_dir_default = './data/evaluation_all/NEUTRAL'
    validation_B_dir_default = './data/evaluation_all/SURPRISE'
    output_dir_default = './validation_output'
    tensorboard_log_dir_default = './log'
    parser.add_argument('--train_A_dir', type = str, help = 'Directory for A.', default = train_A_dir_default)
    parser.add_argument('--train_B_dir', type = str, help = 'Directory for B.', default = train_B_dir_default)
    parser.add_argument('--model_dir', type = str, help = 'Directory for saving models.', default = model_dir_default)
    parser.add_argument('--model_name', type = str, help = 'File name for saving model.', default = model_name_default)
    parser.add_argument('--random_seed', type = int, help = 'Random seed for model training.', default = random_seed_default)
    parser.add_argument('--validation_A_dir', type = str, help = 'Convert validation A after each training epoch. If set none, no conversion would be done during the training.', default = validation_A_dir_default)
    parser.add_argument('--validation_B_dir', type = str, help = 'Convert validation B after each training epoch. If set none, no conversion would be done during the training.', default = validation_B_dir_default)
    parser.add_argument('--output_dir', type = str, help = 'Output directory for converted validation voices.', default = output_dir_default)
    parser.add_argument('--tensorboard_log_dir', type = str, help = 'TensorBoard log directory.', default = tensorboard_log_dir_default)
    args = parser.parse_args()
    
    train(args.train_A_dir, args.train_B_dir, args.model_dir, args.model_name, args.random_seed,
          args.validation_A_dir, args.validation_B_dir, args.output_dir, args.tensorboard_log_dir)
