import os
import torch
import torch.nn as nn
from torch.optim import Adam
from datetime import datetime
from module import *
from utils import l1_loss, l2_loss
import torch.nn.utils as torch_utils
from torch.utils.tensorboard import SummaryWriter

class CycleGAN(nn.Module):
    def __init__(self, num_features, discriminator=discriminator, generator=generator_gatedcnn,
                 mode='train', log_dir='./log', max_grad_norm=10, gp_lambda=5,
                 initial_generator_lr=0.0002, initial_discriminator_lr=0.0001):
        super(CycleGAN, self).__init__()
        
        self.num_features = num_features
        # [batch_size, num_frames, num_features] 
        self.input_shape = [-1, None, num_features]
        #self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.max_grad_norm = max_grad_norm
        # Initialize generators
        self.generator_A2B = generator(num_features)
        self.generator_B2A = generator(num_features)
        
        # Initialize discriminators
        self.discriminator_A = discriminator(num_features)
        self.discriminator_B = discriminator(num_features)
        self.gp_lambda = gp_lambda
        self.mode = mode

        #self.build_model()
        #self.optimizer_initializer()
        self.generator_optimizer = Adam(
            list(self.generator_A2B.parameters()) + list(self.generator_B2A.parameters()),
            lr=initial_generator_lr,  
            betas=(0.5, 0.999)
        )
        self.discriminator_optimizer = Adam(
            list(self.discriminator_A.parameters()) + list(self.discriminator_B.parameters()),
            lr=initial_discriminator_lr,  
            betas=(0.5, 0.999)
        )
        self.generator_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.generator_optimizer, 
            T_0=50,  # restart every 50 epochs
            T_mult=2,  # double the restart interval after each restart
            eta_min=1e-5  # minimum learning rate
        )
        self.discriminator_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.discriminator_optimizer,
            T_0=50,
            T_mult=2,
            eta_min=1e-5
        )
        
        self.scaler = torch.amp.GradScaler('cuda')

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

    '''def compute_generator_losses(self, input_A_real, input_B_real, lambda_cycle, lambda_identity):
        # Generator adversarial losses
        discrimination_B_fake = self.discriminator_B(self.generation_B)
        discrimination_A_fake = self.discriminator_A(self.generation_A)
        
        #self.generator_loss_A2B = torch.mean((discrimination_B_fake - 1) ** 2)
        #self.generator_loss_B2A = torch.mean((discrimination_A_fake - 1) ** 2)

         # Wasserstein loss for generators
        self.generator_loss_A2B = -torch.mean(discrimination_B_fake)
        self.generator_loss_B2A = -torch.mean(discrimination_A_fake)
        
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
        
        return self.generator_loss'''
    
    def compute_gradient_penalty(self, discriminator, real_samples, fake_samples):
        """
        Compute gradient penalty for WGAN-GP
        """
        batch_size = real_samples.size(0)
        
        alpha = torch.rand(batch_size, 1, 1).to(self.device)

        # Interpolate between real and fake samples
        interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)
        # Get discriminator output for interpolated images
        d_interpolates = discriminator(interpolates)

        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones(batch_size, 1).to(self.device),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]

        # Compute gradient penalty
        gradients = gradients.view(batch_size, -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty
        
    '''def compute_discriminator_losses(self, input_A_real, input_B_real, input_A_fake, input_B_fake, gp_lambda):
        
        # Real samples
        self.discrimination_A_real = self.discriminator_A(input_A_real)
        self.discrimination_B_real = self.discriminator_B(input_B_real)
        
        # Fake samples
        self.discrimination_A_fake = self.discriminator_A(input_A_fake.detach())
        self.discrimination_B_fake = self.discriminator_B(input_B_fake.detach())
        
        # Gradient penalty
        gradient_penalty_A = self.compute_gradient_penalty(
            self.discriminator_A, input_A_real, input_A_fake.detach()
        )
        gradient_penalty_B = self.compute_gradient_penalty(
            self.discriminator_B, input_B_real, input_B_fake.detach()
        )

        # Wasserstein loss with gradient penalty
        self.discriminator_loss_A = (
            -torch.mean(self.discrimination_A_real) + 
            torch.mean(self.discrimination_A_fake) +
             gp_lambda * gradient_penalty_A  
        )
        
        self.discriminator_loss_B = (
            -torch.mean(self.discrimination_B_real) + 
            torch.mean(self.discrimination_B_fake) +
            gp_lambda * gradient_penalty_B
        )
        LSE loss:
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
        
        return self.discriminator_loss'''

    def train_step(self, input_A, input_B, lambda_cycle, lambda_identity, n_critic=5):
        # Move inputs to device
        input_A = input_A.to(self.device)
        input_B = input_B.to(self.device)
        # Generate fake samples
        with torch.amp.autocast('cuda'):
            with torch.no_grad():
                generation_A = self.generator_B2A(input_B)
                generation_B = self.generator_A2B(input_A)

        discriminator_loss_total = 0
        discriminator_loss_A_total = 0
        discriminator_loss_B_total = 0
        
        for _ in range(n_critic):       #5 discriminator updates per generator update
            self.discriminator_optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                # Compute discriminator losses
                disc_real_A = self.discriminator_A(input_A)
                disc_fake_A = self.discriminator_A(generation_A.detach())
                disc_real_B = self.discriminator_B(input_B)
                disc_fake_B = self.discriminator_B(generation_B.detach())
                
                # Gradient penalty
                gp_A = self.compute_gradient_penalty(
                    self.discriminator_A, input_A, generation_A.detach()
                )
                gp_B = self.compute_gradient_penalty(
                    self.discriminator_B, input_B, generation_B.detach()
                )
                
                # Wasserstein loss with gradient penalty
                d_loss_A = -torch.mean(disc_real_A) + torch.mean(disc_fake_A) + self.gp_lambda * gp_A
                d_loss_B = -torch.mean(disc_real_B) + torch.mean(disc_fake_B) + self.gp_lambda * gp_B

                discriminator_loss_A_total += d_loss_A.item()
                discriminator_loss_B_total += d_loss_B.item()
                discriminator_loss = d_loss_A + d_loss_B
            
            self.scaler.scale(discriminator_loss).backward()
            torch_utils.clip_grad_norm_(
                list(self.discriminator_A.parameters()) + list(self.discriminator_B.parameters()),
                max_norm=self.max_grad_norm
            )
            self.scaler.step(self.discriminator_optimizer)
            self.scaler.update()

            discriminator_loss_total += discriminator_loss.item()

        # Generator update
        self.generator_optimizer.zero_grad()
        with torch.amp.autocast('cuda'):
            # Forward passes
            generation_A = self.generator_B2A(input_B)
            generation_B = self.generator_A2B(input_A)
            cycle_A = self.generator_B2A(generation_B)
            cycle_B = self.generator_A2B(generation_A)
            identity_A = self.generator_B2A(input_A)
            identity_B = self.generator_A2B(input_B)
            
            # Generator losses
            disc_fake_B = self.discriminator_B(generation_B)
            disc_fake_A = self.discriminator_A(generation_A)
            g_loss_A2B = -torch.mean(disc_fake_B)
            g_loss_B2A = -torch.mean(disc_fake_A)
            
            # Cycle and identity losses
            cycle_loss = (
                torch.mean(torch.abs(input_A - cycle_A)) +
                torch.mean(torch.abs(input_B - cycle_B))
            )
            identity_loss = (
                torch.mean(torch.abs(input_A - identity_A)) +
                torch.mean(torch.abs(input_B - identity_B))
            )
            
            generator_loss = (
                g_loss_A2B +
                g_loss_B2A +
                lambda_cycle * cycle_loss +
                lambda_identity * identity_loss
            )
        
        self.scaler.scale(generator_loss).backward()
        torch_utils.clip_grad_norm_(
            list(self.generator_A2B.parameters()) + list(self.generator_B2A.parameters()),
            max_norm=self.max_grad_norm
        )
        self.scaler.step(self.generator_optimizer)
        self.scaler.update()
        
        # Step the learning rate schedulers
        self.generator_scheduler.step()
        self.discriminator_scheduler.step()

        # Logging
        if self.mode == 'train':
            self.writer.add_scalar('Generator/Total', generator_loss.item(), self.step_count)
            self.writer.add_scalar('Generator/A2B', g_loss_A2B.item(), self.step_count)
            self.writer.add_scalar('Generator/B2A', g_loss_B2A.item(), self.step_count)
            self.writer.add_scalar('Generator/Cycle', cycle_loss.item(), self.step_count)
            self.writer.add_scalar('Generator/Identity', identity_loss.item(), self.step_count)
            self.writer.add_scalar('Discriminator/Total', discriminator_loss_total / n_critic, self.step_count)
            self.writer.add_scalar('Discriminator/A', discriminator_loss_A_total/n_critic, self.step_count)
            self.writer.add_scalar('Discriminator/B', discriminator_loss_B_total/n_critic, self.step_count)
            self.writer.add_scalar('GradientPenalty/A', gp_A.item(), self.step_count)
            self.writer.add_scalar('GradientPenalty/B', gp_B.item(), self.step_count)
            
            # Log learning rates
            self.writer.add_scalar('LearningRate/Generator', 
                                 self.generator_optimizer.param_groups[0]['lr'], 
                                 self.step_count)
            self.writer.add_scalar('LearningRate/Discriminator', 
                                 self.discriminator_optimizer.param_groups[0]['lr'], 
                                 self.step_count)
            
            self.step_count += 1
            
        return generator_loss.item(), discriminator_loss_total / n_critic
    
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
            self.step_count = checkpoint['step_count']

if __name__ == '__main__':
    model = CycleGAN(num_features=24, discriminator=discriminator, generator=generator_gatedcnn)
    print('Model initialized successfully.')
