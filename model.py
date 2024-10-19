import os
import torch
import torch.nn as nn
from datetime import datetime
from module import Discriminator, GeneratorGatedCNN
from utils import l1_loss, l2_loss

class CycleGAN:
    def __init__(self, num_features, discriminator=Discriminator, generator=GeneratorGatedCNN, mode='train', log_dir='./log'):
        self.num_features = num_features
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.discriminator = discriminator().to(self.device)
        self.generator = generator().to(self.device)
        self.mode = mode

        self.build_model()
        self.optimizer_initializer()

        if self.mode == 'train':
            self.train_step = 0
            now = datetime.now()
            self.log_dir = os.path.join(log_dir, now.strftime('%Y%m%d-%H%M%S'))
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(self.log_dir)

    def build_model(self):
        # Tensors for real and fake samples
        self.input_A_real = torch.zeros((1, self.num_features, 256)).to(self.device)
        self.input_B_real = torch.zeros((1, self.num_features, 256)).to(self.device)
        self.input_A_fake = torch.zeros((1, self.num_features, 256)).to(self.device)
        self.input_B_fake = torch.zeros((1, self.num_features, 256)).to(self.device)

        # Generator forward passes
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

        # Cycle loss
        self.cycle_loss = l1_loss(self.input_A_real, self.cycle_A) + l1_loss(self.input_B_real, self.cycle_B)

        # Identity loss
        self.identity_loss = l1_loss(self.input_A_real, self.generation_A_identity) + l1_loss(self.input_B_real, self.generation_B_identity)

        # Generator loss (attempting to fool the discriminator)
        self.generator_loss_A2B = l2_loss(torch.ones_like(self.discrimination_B_fake), self.discrimination_B_fake)
        self.generator_loss_B2A = l2_loss(torch.ones_like(self.discrimination_A_fake), self.discrimination_A_fake)
        self.generator_loss = self.generator_loss_A2B + self.generator_loss_B2A + self.lambda_cycle * self.cycle_loss + self.lambda_identity * self.identity_loss

        # Discriminator loss (distinguishing real and fake)
        self.discrimination_input_A_real = self.discriminator(self.input_A_real)
        self.discrimination_input_B_real = self.discriminator(self.input_B_real)
        self.discrimination_input_A_fake = self.discriminator(self.input_A_fake)
        self.discrimination_input_B_fake = self.discriminator(self.input_B_fake)

        self.discriminator_loss_A_real = l2_loss(torch.ones_like(self.discrimination_input_A_real), self.discrimination_input_A_real)
        self.discriminator_loss_A_fake = l2_loss(torch.zeros_like(self.discrimination_input_A_fake), self.discrimination_input_A_fake)
        self.discriminator_loss_A = (self.discriminator_loss_A_real + self.discriminator_loss_A_fake) / 2

        self.discriminator_loss_B_real = l2_loss(torch.ones_like(self.discrimination_input_B_real), self.discrimination_input_B_real)
        self.discriminator_loss_B_fake = l2_loss(torch.zeros_like(self.discrimination_input_B_fake), self.discrimination_input_B_fake)
        self.discriminator_loss_B = (self.discriminator_loss_B_real + self.discriminator_loss_B_fake) / 2

        self.discriminator_loss = self.discriminator_loss_A + self.discriminator_loss_B

    def optimizer_initializer(self):
        # Optimizer initialization
        self.generator_optimizer = torch.optim.Adam(self.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.discriminator_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    def train(self, input_A, input_B, lambda_cycle, lambda_identity, generator_lr, discriminator_lr):
        # Set learning rates
        for param_group in self.generator_optimizer.param_groups:
            param_group['lr'] = generator_lr
        for param_group in self.discriminator_optimizer.param_groups:
            param_group['lr'] = discriminator_lr

        # Forward pass and compute losses
        self.generation_A = self.generator(input_B)
        self.generation_B = self.generator(input_A)
        cycle_loss = l1_loss(input_A, self.generator(self.generation_B)) + l1_loss(input_B, self.generator(self.generation_A))
        identity_loss = l1_loss(input_A, self.generator(input_A)) + l1_loss(input_B, self.generator(input_B))
        generator_loss = cycle_loss + lambda_cycle * cycle_loss + lambda_identity * identity_loss

        # Optimize the generator
        self.generator_optimizer.zero_grad()
        generator_loss.backward()
        self.generator_optimizer.step()

        # Discriminator loss
        discriminator_loss_A = l2_loss(torch.ones_like(self.discriminator(input_A)), self.discriminator(self.generation_A)) + \
                               l2_loss(torch.zeros_like(self.discriminator(input_B)), self.discriminator(self.generation_B))
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
                raise ValueError("Conversion direction must be 'A2B' or 'B2A'.")

    def save(self, directory, filename):
        if not os.path.exists(directory):
            os.makedirs(directory)
        torch.save({
            'generator': self.generator.state_dict(),
            'discriminator': self.discriminator.state_dict(),
            'generator_optimizer': self.generator_optimizer.state_dict(),
            'discriminator_optimizer': self.discriminator_optimizer.state_dict()
        }, os.path.join(directory, filename))

    def load(self, filepath):
        checkpoint = torch.load(filepath)
        self.generator.load_state_dict(checkpoint['generator'])
        self.discriminator.load_state_dict(checkpoint['discriminator'])
        self.generator_optimizer.load_state_dict(checkpoint['generator_optimizer'])
        self.discriminator_optimizer.load_state_dict(checkpoint['discriminator_optimizer'])

    def summary(self):
        return None  # Placeholder for future tensorboard integration

if __name__ == '__main__':
    model = CycleGAN(num_features=24)
    print('Model built successfully.')
