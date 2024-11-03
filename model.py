import os
import torch
import torch.nn as nn
from torch.optim import Adam
from datetime import datetime
from module import *
from utils import l1_loss, l2_loss
from torch.utils.tensorboard import SummaryWriter

class CycleGAN:
    def __init__(self, num_features, discriminator=discriminator, generator=generator_gatedcnn, mode='train', log_dir='./log'):
        super(CycleGAN, self).__init__()
        self.num_features = num_features
        # [batch_size, num_frames, num_features] 
        self.input_shape = [-1, None, num_features]
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Initialize generators
        self.generator_A2B = generator(num_features)
        self.generator_B2A = generator(num_features)
        
        # Initialize discriminators
        self.discriminator_A = discriminator(num_features)
        self.discriminator_B = discriminator(num_features)
       
        self.mode = mode

        #self.build_model()
        #self.optimizer_initializer()

        if self.mode == 'train':
            self.step_count = 0
            now = datetime.now()
            self.log_dir = os.path.join(log_dir, now.strftime('%Y%m%d-%H%M%S'))
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(self.log_dir)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, input_A, input_B):
        # Generator forward passes
        self.generation_B = self.generator_A2B(input_A)
        self.cycle_A = self.generator_B2A(self.generation_B)

        self.generation_A = self.generator_B2A(input_B)
        self.cycle_B = self.generator_A2B(self.generation_A)

        # Identity mapping
        self.generation_A_identity = self.generator_B2A(input_A)
        self.generation_B_identity = self.generator_A2B(input_B)
        
        return self.generation_A, self.generation_B

    def compute_generator_losses(self, input_A_real, input_B_real, lambda_cycle, lambda_identity):
        # Generator adversarial losses
        discrimination_B_fake = self.discriminator_B(self.generation_B)
        discrimination_A_fake = self.discriminator_A(self.generation_A)
        
        self.generator_loss_A2B = torch.mean((discrimination_B_fake - 1) ** 2)
        self.generator_loss_B2A = torch.mean((discrimination_A_fake - 1) ** 2)
        
        # Cycle consistency losses
        self.cycle_loss = (
            torch.mean(torch.abs(input_A_real - self.cycle_A)) +
            torch.mean(torch.abs(input_B_real - self.cycle_B))
        )
        
        # Identity losses
        self.identity_loss = (
            torch.mean(torch.abs(input_A_real - self.generation_A_identity)) +
            torch.mean(torch.abs(input_B_real - self.generation_B_identity))
        )
        
        # Total generator loss
        self.generator_loss = (
            self.generator_loss_A2B +
            self.generator_loss_B2A +
            lambda_cycle * self.cycle_loss +
            lambda_identity * self.identity_loss
        )
        
        return self.generator_loss
        
    def compute_discriminator_losses(self, input_A_real, input_B_real, input_A_fake, input_B_fake):
        # Real samples
        self.discrimination_A_real = self.discriminator_A(input_A_real)
        self.discrimination_B_real = self.discriminator_B(input_B_real)
        
        # Fake samples
        self.discrimination_A_fake = self.discriminator_A(input_A_fake.detach())
        self.discrimination_B_fake = self.discriminator_B(input_B_fake.detach())
        
        # Discriminator losses
        self.discriminator_loss_A = (
            torch.mean((self.discrimination_A_real - 1) ** 2) +
            torch.mean(self.discrimination_A_fake ** 2)
        ) / 2
        
        self.discriminator_loss_B = (
            torch.mean((self.discrimination_B_real - 1) ** 2) +
            torch.mean(self.discrimination_B_fake ** 2)
        ) / 2
        
        # Total discriminator loss
        self.discriminator_loss = self.discriminator_loss_A + self.discriminator_loss_B
        
        return self.discriminator_loss

    def train_step(self, input_A, input_B, lambda_cycle, lambda_identity, generator_learning_rate, discriminator_learning_rate):
        # Move inputs to device
        input_A = input_A.to(self.device)
        input_B = input_B.to(self.device)
        
        # Initialize optimizers
        generator_optimizer = Adam(
            list(self.generator_A2B.parameters()) + list(self.generator_B2A.parameters()),
            lr=generator_learning_rate,
            betas=(0.5, 0.999)
        )
        discriminator_optimizer = Adam(
            list(self.discriminator_A.parameters()) + list(self.discriminator_B.parameters()),
            lr=discriminator_learning_rate,
            betas=(0.5, 0.999)
        )
         # Generator forward pass and loss computation
        generator_optimizer.zero_grad()
        generation_A, generation_B = self(input_A, input_B)
        generator_loss = self.compute_generator_losses(input_A, input_B, lambda_cycle, lambda_identity)
        generator_loss.backward()
        generator_optimizer.step()
        
        # Discriminator forward pass and loss computation
        discriminator_optimizer.zero_grad()
        discriminator_loss = self.compute_discriminator_losses(input_A, input_B, generation_A, generation_B)
        discriminator_loss.backward()
        discriminator_optimizer.step()
        
        # Log losses
        if self.mode == 'train':
            self.writer.add_scalar('Generator/Total', generator_loss.item(), self.step_count)
            self.writer.add_scalar('Generator/A2B', self.generator_loss_A2B.item(), self.step_count)
            self.writer.add_scalar('Generator/B2A', self.generator_loss_B2A.item(), self.step_count)
            self.writer.add_scalar('Generator/Cycle', self.cycle_loss.item(), self.step_count)
            self.writer.add_scalar('Generator/Identity', self.identity_loss.item(), self.step_count)
            self.writer.add_scalar('Discriminator/Total', discriminator_loss.item(), self.step_count)
            self.writer.add_scalar('Discriminator/A', self.discriminator_loss_A.item(), self.step_count)
            self.writer.add_scalar('Discriminator/B', self.discriminator_loss_B.item(), self.step_count)
            self.step_count += 1
            
        return generator_loss.item(), discriminator_loss.item()
    
    @torch.no_grad()   
    def test(self, inputs, direction):
        inputs = inputs.to(self.device)
        if direction == 'A2B':
            generation = self.generator_A2B(inputs)
        elif direction == 'B2A':
            generation = self.generator_B2A(inputs)
        else:
            raise ValueError('Conversion direction must be specified.')
        return generation

    def save(self, directory, filename):
        if not os.path.exists(directory):
            os.makedirs(directory)
        filepath = os.path.join(directory, filename)
        torch.save({
            'generator_A2B_state_dict': self.generator_A2B.state_dict(),
            'generator_B2A_state_dict': self.generator_B2A.state_dict(),
            'discriminator_A_state_dict': self.discriminator_A.state_dict(),
            'discriminator_B_state_dict': self.discriminator_B.state_dict(),
            'step_count': self.step_count if self.mode == 'train' else 0  
        }, filepath)
        return filepath

    def load(self, filepath):
        checkpoint = torch.load(filepath, map_location=self.device)
        self.generator_A2B.load_state_dict(checkpoint['generator_A2B_state_dict'])
        self.generator_B2A.load_state_dict(checkpoint['generator_B2A_state_dict'])
        self.discriminator_A.load_state_dict(checkpoint['discriminator_A_state_dict'])
        self.discriminator_B.load_state_dict(checkpoint['discriminator_B_state_dict'])
        if self.mode == 'train':
            self.train_step = checkpoint['step_count']

if __name__ == '__main__':
    model = CycleGAN(num_features=24, discriminator=discriminator, generator=generator_gatedcnn)
    print('Model initialized successfully.')
