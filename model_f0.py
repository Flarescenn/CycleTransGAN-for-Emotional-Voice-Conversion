import torch
import torch.nn as nn
import os
from datetime import datetime
from module_f0 import Discriminator, GeneratorGatedCNN
from utils import l1_loss, l2_loss

class CycleGAN:
    def __init__(self, num_features, discriminator=Discriminator, generator=GeneratorGatedCNN, mode='train', log_dir='./log'):
        self.num_features = num_features
        self.discriminator = discriminator().to(self.device)
        self.generator = generator().to(self.device)
        self.mode = mode
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.build_model()
        self.optimizer_initializer()

        if self.mode == 'train':
            self.train_step = 0
            now = datetime.now()
            self.log_dir = os.path.join(log_dir, now.strftime('%Y%m%d-%H%M%S'))
            # TensorBoard setup
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(self.log_dir)

    def build_model(self):
        # Placeholder for real and fake training samples
        self.input_A_real = torch.zeros((1, self.num_features, 256)).to(self.device)
        self.input_B_real = torch.zeros((1, self.num_features, 256)).to(self.device)
        self.input_A_fake = torch.zeros((1, self.num_features, 256)).to(self.device)
        self.input_B_fake = torch.zeros((1, self.num_features, 256)).to(self.device)

        # Forward passes through the generator
        self.generation_B = self.generator(self.input_A_real)
        self.cycle_A = self.generator(self.generation_B)

        self.generation_A = self.generator(self.input_B_real)
        self.cycle_B = self.generator(self.generation_A)

        # Identity generation
        self.generation_A_identity = self.generator(self.input_A_real)
        self.generation_B_identity = self.generator(self.input_B_real)

        # Discriminator forward passes
        self.discrimination_A_fake = self.discriminator(self.generation_A)
        self.discrimination_B_fake = self.discriminator(self.generation_B)

        # Cycle and identity losses
        self.cycle_loss = l1_loss(self.input_A_real, self.cycle_A) + l1_loss(self.input_B_real, self.cycle_B)
        self.identity_loss = l1_loss(self.input_A_real, self.generation_A_identity) + l1_loss(self.input_B_real, self.generation_B_identity)

    def optimizer_initializer(self):
        # Learning rates
        self.generator_lr = 2e-4
        self.discriminator_lr = 2e-4

        # Optimizers
        self.generator_optimizer = torch.optim.Adam(self.generator.parameters(), lr=self.generator_lr, betas=(0.5, 0.999))
        self.discriminator_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=self.discriminator_lr, betas=(0.5, 0.999))

    def train(self, input_A, input_B, lambda_cycle, lambda_identity):
        self.generator_optimizer.zero_grad()
        self.discriminator_optimizer.zero_grad()

        # Forward pass through the generators
        self.generation_A = self.generator(input_B)
        self.generation_B = self.generator(input_A)

        # Cycle and identity losses
        cycle_loss = l1_loss(input_A, self.generator(self.generation_B)) + l1_loss(input_B, self.generator(self.generation_A))
        identity_loss = l1_loss(input_A, self.generator(input_A)) + l1_loss(input_B, self.generator(input_B))

        # Generator loss
        generator_loss = cycle_loss + lambda_cycle * cycle_loss + lambda_identity * identity_loss
        generator_loss.backward()
        self.generator_optimizer.step()

        # Discriminator forward pass
        discriminator_loss_A = l2_loss(torch.ones_like(self.discriminator(input_A)), self.discriminator(self.generation_A)) + l2_loss(torch.zeros_like(self.discriminator(input_B)), self.discriminator(self.generation_B))
        discriminator_loss_A.backward()
        self.discriminator_optimizer.step()

        return generator_loss, discriminator_loss_A

    def test(self, inputs, direction):
        with torch.no_grad():
            if direction == 'A2B':
                return self.generator(inputs)
            elif direction == 'B2A':
                return self.generator(inputs)
            else:
                raise ValueError('Conversion direction must be either A2B or B2A.')

    def save(self, directory, filename):
        if not os.path.exists(directory):
            os.makedirs(directory)
        torch.save({
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'optimizerG_state_dict': self.generator_optimizer.state_dict(),
            'optimizerD_state_dict': self.discriminator_optimizer.state_dict(),
        }, os.path.join(directory, filename))

    def load(self, filepath):
        checkpoint = torch.load(filepath)
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        self.generator_optimizer.load_state_dict(checkpoint['optimizerG_state_dict'])
        self.discriminator_optimizer.load_state_dict(checkpoint['optimizerD_state_dict'])

if __name__ == '__main__':
    model = CycleGAN(num_features=10)
    print('Model built successfully.')
