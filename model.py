import os
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from datetime import datetime
from module import *
from torch.utils.tensorboard import SummaryWriter

class CycleGAN(nn.Module):
    def __init__(self, num_features, initial_generator_lr, initial_discriminator_lr,
                 mode='train', discriminator=discriminator, generator=generator_gatedcnn,
                 log_dir='./log'):
        super(CycleGAN, self).__init__()
        self.num_features = num_features
        self.mode = mode
        self.initial_generator_lr = initial_generator_lr
        self.initial_discriminator_lr = initial_discriminator_lr
        # Initialize networks
        self.generator_A2B = generator(num_features)
        self.generator_B2A = generator(num_features)
        self.discriminator_A = discriminator(num_features)
        self.discriminator_B = discriminator(num_features)
        
        # Initialize optimizers
        self.generator_optimizer = Adam(
            list(self.generator_A2B.parameters()) + 
            list(self.generator_B2A.parameters()),
            lr=initial_generator_lr, betas=(0.5, 0.999)
        )
        self.discriminator_optimizer = Adam(
            list(self.discriminator_A.parameters()) + 
            list(self.discriminator_B.parameters()),
            lr=initial_discriminator_lr, betas=(0.5, 0.999)
        )

        self.generator_scheduler = None
        self.discriminator_scheduler = None
        '''# Initialize schedulers
        self.generator_scheduler = CosineAnnealingWarmRestarts(
            self.generator_optimizer, T_0=50000, T_mult=2, eta_min=1e-6
        )
        self.discriminator_scheduler = CosineAnnealingWarmRestarts(
            self.discriminator_optimizer, T_0=50000, T_mult=2, eta_min=1e-7
        )
        '''
        # Setup logging for training mode
        if self.mode == 'train':
            self.step_count = 0
            now = datetime.now()
            self.log_dir = os.path.join(log_dir, now.strftime('%Y%m%d-%H%M%S'))
            self.writer = SummaryWriter(self.log_dir)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def setup_schedulers(self, dataloader, num_epochs): #For setting up the OneCycleLR
        total_steps = len(dataloader) * num_epochs
        print(f"Configuring OneCycleLR with total_steps: {total_steps}")
        
        self.generator_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.generator_optimizer,
            max_lr=self.initial_generator_lr,
            total_steps=total_steps,
            pct_start=0.3,
            anneal_strategy='cos',
            div_factor=25,
            final_div_factor=100
        )
        
        self.discriminator_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.discriminator_optimizer,
            max_lr=self.initial_discriminator_lr,
            total_steps=total_steps,
            pct_start=0.3,
            anneal_strategy='cos',
            div_factor=25,
            final_div_factor=100
        )

    '''def compute_gradient_penalty(self, discriminator, real_samples, fake_samples):
        batch_size = real_samples.size(0)
        alpha = torch.rand(batch_size, 1, 1).to(self.device)
        
        # Interpolate between real and fake samples
        interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)
        d_interpolates = discriminator(interpolates)
        
        # Compute gradients
        fake = torch.ones(d_interpolates.size()).to(self.device)
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        
        # Calculate gradient penalty
        gradients = gradients.view(batch_size, -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty'''

    def forward(self, input_A, input_B):
        self.real_A = input_A
        self.real_B = input_B
        
        # Generator forward passes
        self.fake_B = self.generator_A2B(self.real_A)
        self.cycle_A = self.generator_B2A(self.fake_B)
        
        self.fake_A = self.generator_B2A(self.real_B)
        self.cycle_B = self.generator_A2B(self.fake_A)
        
        # Identity mapping
        self.identity_A = self.generator_B2A(self.real_A)
        self.identity_B = self.generator_A2B(self.real_B)
        
        return self.fake_A, self.fake_B

    def compute_generator_losses(self, lambda_cycle, lambda_identity):
        # Adversarial losses (Wasserstein)
        self.loss_G_A2B = -torch.mean(self.discriminator_B(self.fake_B))
        self.loss_G_B2A = -torch.mean(self.discriminator_A(self.fake_A))
        
        # Cycle consistency losses
        self.loss_cycle = (
            torch.mean(torch.abs(self.real_A - self.cycle_A)) +
            torch.mean(torch.abs(self.real_B - self.cycle_B))
        )
        
        # Identity losses
        self.loss_identity = (
            torch.mean(torch.abs(self.real_A - self.identity_A)) +
            torch.mean(torch.abs(self.real_B - self.identity_B))
        )
        
        # Total generator loss
        self.loss_G = (
            self.loss_G_A2B +
            self.loss_G_B2A +
            lambda_cycle * self.loss_cycle +
            lambda_identity * self.loss_identity
        )
        
        return self.loss_G

    def compute_discriminator_losses(self):
        # Wasserstein losses without gradient penalty
        self.loss_D_A = (
            -torch.mean(self.discriminator_A(self.real_A)) +
            torch.mean(self.discriminator_A(self.fake_A.detach()))
            )
        
        
        self.loss_D_B = (
            -torch.mean(self.discriminator_B(self.real_B)) +
            torch.mean(self.discriminator_B(self.fake_B.detach())))          
        
        
        self.loss_D = self.loss_D_A + self.loss_D_B
        return self.loss_D

    def train_step(self, input_A, input_B, lambda_cycle, lambda_identity):
        # Move inputs to device
        self.real_A = input_A.to(self.device)
        self.real_B = input_B.to(self.device)
        
        # Generator forward and backward pass
        self.generator_optimizer.zero_grad()
        self(self.real_A, self.real_B)  # Forward pass
        loss_G = self.compute_generator_losses(lambda_cycle, lambda_identity)
        loss_G.backward()
        self.generator_optimizer.step()
        
        # Discriminator forward and backward pass
        self.discriminator_optimizer.zero_grad()
        loss_D = self.compute_discriminator_losses()
        loss_D.backward()
        self.discriminator_optimizer.step()
        
        # Update learning rates
        self.generator_scheduler.step()
        self.discriminator_scheduler.step()
        
        # Log losses
        if self.mode == 'train':
            self.writer.add_scalar('Generator/Total', loss_G.item(), self.step_count)
            self.writer.add_scalar('Generator/A2B', self.loss_G_A2B.item(), self.step_count)
            self.writer.add_scalar('Generator/B2A', self.loss_G_B2A.item(), self.step_count)
            self.writer.add_scalar('Generator/Cycle', self.loss_cycle.item(), self.step_count)
            self.writer.add_scalar('Generator/Identity', self.loss_identity.item(), self.step_count)
            self.writer.add_scalar('Discriminator/Total', loss_D.item(), self.step_count)
            self.writer.add_scalar('Discriminator/A', self.loss_D_A.item(), self.step_count)
            self.writer.add_scalar('Discriminator/B', self.loss_D_B.item(), self.step_count)
            self.step_count += 1
        
        return loss_G.item(), loss_D.item()

    @torch.no_grad()
    def test(self, inputs, direction):
        inputs = inputs.to(self.device)
        if direction == 'A2B':
            return self.generator_A2B(inputs)
        elif direction == 'B2A':
            return self.generator_B2A(inputs)
        else:
            raise ValueError('Direction must be either A2B or B2A')

    def save(self, directory, filename):
        if not os.path.exists(directory):
            os.makedirs(directory)
        filepath = os.path.join(directory, filename)
        torch.save({
            'generator_A2B_state_dict': self.generator_A2B.state_dict(),
            'generator_B2A_state_dict': self.generator_B2A.state_dict(),
            'discriminator_A_state_dict': self.discriminator_A.state_dict(),
            'discriminator_B_state_dict': self.discriminator_B.state_dict(),
            'generator_optimizer_state_dict': self.generator_optimizer.state_dict(),
            'discriminator_optimizer_state_dict': self.discriminator_optimizer.state_dict(),
            'generator_scheduler_state_dict': self.generator_scheduler.state_dict(),
            'discriminator_scheduler_state_dict': self.discriminator_scheduler.state_dict(),
            'step_count': self.step_count if self.mode == 'train' else 0
        }, filepath)
        return filepath

    def load(self, filepath):
        checkpoint = torch.load(filepath, map_location=self.device)
        self.generator_A2B.load_state_dict(checkpoint['generator_A2B_state_dict'])
        self.generator_B2A.load_state_dict(checkpoint['generator_B2A_state_dict'])
        self.discriminator_A.load_state_dict(checkpoint['discriminator_A_state_dict'])
        self.discriminator_B.load_state_dict(checkpoint['discriminator_B_state_dict'])
        self.generator_optimizer.load_state_dict(checkpoint['generator_optimizer_state_dict'])
        self.discriminator_optimizer.load_state_dict(checkpoint['discriminator_optimizer_state_dict'])
        self.generator_scheduler.load_state_dict(checkpoint['generator_scheduler_state_dict'])
        self.discriminator_scheduler.load_state_dict(checkpoint['discriminator_scheduler_state_dict'])
        if self.mode == 'train':
            self.step_count = checkpoint['step_count']