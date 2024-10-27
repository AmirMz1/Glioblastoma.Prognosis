from octavecgan import *
import torch
import torch.optim as optim
import torch.nn as nn
import torch
import torch.autograd as autograd
torch.autograd.set_detect_anomaly(True)
import argparse

from torch.utils.data import DataLoader, TensorDataset, Dataset
from sklearn.model_selection import train_test_split
import torchvision.utils as vutils

import matplotlib.pyplot as plt
import gc

import torch.nn.functional as F
import math

def load_checkpoint(netG, netD, optimizerG, optimizerD):
    print(f"Loading generator checkpoint")
    netG.load_state_dict(torch.load(f'weights/best_octave_unet_generator{args.vol}.pth', weights_only=True))
    netD.load_state_dict(torch.load(f'weights/best_octave_unet_discriminator{args.vol}.pth', weights_only=True))



class MRIDataset(Dataset):
    def __init__(self, t1_images_path, flair_images_path, t2_images_path, labels_path, indices=None):
        self.t1_images = np.load(t1_images_path, mmap_mode='r')
        self.flair_images = np.load(flair_images_path, mmap_mode='r')
        self.t2_images = np.load(t2_images_path, mmap_mode='r')
        self.labels = np.load(labels_path, mmap_mode='r')
        self.indices = indices if indices is not None else range(len(self.labels))
        
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        t1 = torch.tensor(self.t1_images[real_idx], dtype=torch.float32).unsqueeze(0)
        flair = torch.tensor(self.flair_images[real_idx], dtype=torch.float32).unsqueeze(0)
        t2 = torch.tensor(self.t2_images[real_idx], dtype=torch.float32).unsqueeze(0)
        label = torch.tensor(self.labels[real_idx], dtype=torch.long)
        return torch.cat((t1, flair), dim=0), t2, label

def load_dataset(t1_images, flair_images, t2_images, labels):
    indices = np.arange(len(np.load(labels, mmap_mode='r')))
    train_indices, val_indices = train_test_split(indices, test_size=0.2, random_state=42, stratify=np.load(labels, mmap_mode='r'))
    
    train_dataset = MRIDataset(t1_images, flair_images, t2_images, labels, indices=train_indices)
    val_dataset = MRIDataset(t1_images, flair_images, t2_images, labels, indices=val_indices)

    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, pin_memory=True, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False, pin_memory=True, num_workers=4)

    return train_dataloader, val_dataloader



# def load_dataset(t1_images, flair_images, t2_images, labels):
#     t1_images = np.load(t1_images)
#     flair_images = np.load(flair_images)
#     t2_images = np.load(t2_images)
#     labels = np.load(labels)

#     # تقسیم داده‌ها به مجموعه‌های آموزشی و اعتبارسنجی
#     t1_train, t1_val, flair_train, flair_val, t2_train, t2_val, labels_train, labels_val = train_test_split(
#         t1_images, flair_images, t2_images, labels, test_size=0.2, random_state=42, stratify=labels
#     )

#     # Clear loaded dataset from memory
#     del t1_images, flair_images, t2_images, labels
#     gc.collect()

#     # تبدیل داده‌ها به تنسورهای PyTorch بدون انتقال به دستگاه مورد نظر
#     t1_train, flair_train, t2_train = map(lambda x: torch.tensor(x, dtype=torch.float32).unsqueeze(1), [t1_train, flair_train, t2_train])
#     t1_val, flair_val, t2_val = map(lambda x: torch.tensor(x, dtype=torch.float32).unsqueeze(1), [t1_val, flair_val, t2_val])
#     labels_train, labels_val = torch.tensor(labels_train, dtype=torch.long), torch.tensor(labels_val, dtype=torch.long)

#     # ترکیب کردن T1 و Flair به عنوان ورودی‌ها و استفاده از T2 به عنوان تصاویر هدف
#     train_inputs = torch.cat((t1_train, flair_train), dim=1)
#     val_inputs = torch.cat((t1_val, flair_val), dim=1)

#     # ساختار TensorDataset همراه با labels_train و labels_val
#     train_dataset = TensorDataset(train_inputs, t2_train, labels_train)
#     val_dataset = TensorDataset(val_inputs, t2_val, labels_val)

#     # ایجاد DataLoader برای مجموعه‌های آموزشی و اعتبارسنجی بدون prefetch_factor
#     train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, pin_memory=True, num_workers=0)
#     val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False, pin_memory=True, num_workers=0)


#     # چاپ اطلاعات مجموعه‌ها
#     print("Training set size:", len(t1_train))
#     print("Validation set size:", len(t1_val))

#     unique, counts = np.unique(labels_train.numpy(), return_counts=True)
#     print("Training set label distribution:", dict(zip(unique, counts)))

#     unique, counts = np.unique(labels_val.numpy(), return_counts=True)
#     print("Validation set label distribution:", dict(zip(unique, counts)))
#     print('\n\n')

#     # Clear intermediate variables
#     del t1_train, flair_train, t2_train, t1_val, flair_val, t2_val
#     gc.collect()

#     # تابع بررسی وجود NaN   یا Inf در داده‌ها
#     def check_for_nan_inf(tensor, tensor_name):
#         if torch.isnan(tensor).any():
#             print(f"NaN detected in {tensor_name}")
#         if torch.isinf(tensor).any():
#             print(f"Inf detected in {tensor_name}")

#     # حالا، در هنگام پردازش هر batch داده‌ها را به TPU منتقل می‌کنیم
#     for inputs, targets, labels in train_dataloader:
#         # بررسی داده‌های ورودی به TPU
#         check_for_nan_inf(inputs, "inputs before TPU")
#         check_for_nan_inf(targets, "targets before TPU")
#         check_for_nan_inf(labels, "labels before TPU")

#         # انتقال داده‌ها به دستگاه TPU
#         inputs = inputs.to(device)
#         targets = targets.to(device)
#         labels = labels.to(device)

#     # Clear cache and collect garbage
#     torch.cuda.empty_cache()
#     gc.collect()

#     return train_dataloader, val_dataloader


# WGAN loss function (simple difference)
# WGAN loss function (simple difference)
def wgan_discriminator_loss(real_output, fake_output, gp=None, lambda_gp=None):
    # return (torch.mean(fake_output) - torch.mean(real_output)) + lambda_gp * gp
    return -torch.mean(real_output) + torch.mean(fake_output)

def wgan_generator_loss(fake_output):
    return -torch.mean(fake_output)


# Gradient penalty for WGAN-GP
def gradient_penalty(netD, real_data, fake_data, device='cuda'):
    batch_size = real_data.size(0)
    epsilon = torch.rand(batch_size, 1, 1, 1, device=device, requires_grad=True)
    interpolated = epsilon * real_data + (1 - epsilon) * fake_data
    interpolated = interpolated.to(device)
    
    interpolated_output = netD(interpolated)
    gradients = autograd.grad(
        outputs=interpolated_output, inputs=interpolated,
        grad_outputs=torch.ones(interpolated_output.size(), device=device),
        create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradients = gradients.view(batch_size, -1)
    gradient_norm = gradients.norm(2, dim=1)
    penalty = torch.mean((gradient_norm - 1) ** 2)
    return penalty

def calculate_psnr(real, fake, max_pixel_value=1.0):
    mse = F.mse_loss(fake, real)
    if mse == 0:  # MSE can be zero if the images are identical
        return float('inf')
    psnr = 20 * math.log10(max_pixel_value / math.sqrt(mse.item()))
    return psnr
    
def train(train_dataloader, val_dataloader, use_gp=True):
    num_epochs = args.n_epochs
    best_val_loss = float('inf')
    min_delta = 1e-4
    patience = 5
    best_psnr = 0
    # n_critic = 5
    # clip_value = 0.01 
    early_stop_counter = 0

    # OctaveUnet / Generator
    netG = define_G(input_nc=2, output_nc=1, norm='batch', use_dropout=False, gpu_ids=[])
    netG = netG.to(device)

    # Discriminator
    netD = define_D(input_nc=1, ndf=64, norm='batch', use_sigmoid=False, gpu_ids=[])  # No sigmoid in WGAN
    netD = netD.to(device)

    optimizerG = optim.Adam(netG.parameters(), lr=1e-4, betas=(0.5, 0.999))
    optimizerD = optim.Adam(netD.parameters(), lr=1e-5, betas=(0.5, 0.999))


    # Load checkpoints if resuming training
    if args.resume:
        load_checkpoint(netG, netD, optimizerG, optimizerD)
        
    # lambda_gp = 10  # Gradient penalty weight (for WGAN-GP)

    G_losses = []
    D_losses = []
    psnr_scores = []
    
    for epoch in range(num_epochs):
        netG.train()
        netD.train()

        for i, data in enumerate(train_dataloader, 0):
            # ------------------------------
            # Training the Discriminator
            # ------------------------------
            netD.zero_grad()
            real_cpu = data[1].to(device).float()
            b_size = real_cpu.size(0)

            # Fake data
            fake = netG(data[0].to(device).float())

            # Gradient penalty for WGAN-GP
            # gp = gradient_penalty(netD, real_cpu, fake, device)
            # Wasserstein loss for discriminator
            errD_real = wgan_discriminator_loss(netD(real_cpu), netD(fake.detach()))
    
            # errD_real.backward(retain_graph=True)
            errD_real.backward()
            optimizerD.step()


            # ------------------------------
            # Training the Generator
            # ------------------------------
            netG.zero_grad()

            # Wasserstein loss for generator
            errG = wgan_generator_loss(netD(fake))
            errG.backward()

            optimizerG.step()

            G_losses.append(errG.item())
            D_losses.append(errD_real.item())

            # printing the results
            if i % 100 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f'
                      % (epoch+1, num_epochs, i, len(train_dataloader), errD_real.item(), errG.item()))

            if torch.isnan(errG) or torch.isnan(errD_real):
                print(f'NaN detected at iteration {i}')
                break

            # del real_cpu, fake, real_output, fake_output, errD_real, errG
            torch.cuda.empty_cache()

        val_loss = 0.0
        total_psnr = 0.0
        netG.eval()
        with torch.no_grad():
            for i, data in enumerate(val_dataloader, 0):
                real_cpu = data[1].to(device).float()
                fake = netG(data[0].to(device).float())

                fake_output = netD(fake)
                val_loss += wgan_generator_loss(fake_output).item()

                # Calculate PSNR
                psnr = calculate_psnr(real_cpu, fake)
                total_psnr += psnr

        avg_val_loss = val_loss / len(val_dataloader)
        avg_psnr = total_psnr / len(val_dataloader)
        psnr_scores.append(avg_psnr)

        print(f'Epoch [{epoch+1}/{num_epochs}], Validation Loss: {avg_val_loss:.4f}, PSNR: {avg_psnr:.2f} dB')

        # Save best models based on PSNR
        if avg_psnr > best_psnr:
            best_psnr = avg_psnr
            early_stop_counter = 0
            torch.save(netG.state_dict(), f'weights/best_octave_unet_generator{args.vol}.pth')
            torch.save(netD.state_dict(), f'weights/best_octave_unet_discriminator{args.vol}.pth')
        else:
            early_stop_counter += 1


        # if early_stop_counter >= patience:
            # print("Early stopping triggered.")
            # break

        # # Save best models based on PSNR
        # if avg_psnr > best_psnr:
        #     best_psnr = avg_psnr
        #     early_stop_counter = 0
        #     torch.save(netG.state_dict(), f'weights/best_octave_unet_generator{args.vol}.pth')
        #     torch.save(netD.state_dict(), f'weights/best_octave_unet_discriminator{args.vol}.pth')
        # else:
        #     early_stop_counter += 1

        # if early_stop_counter >= patience:
        #     print("Early stopping triggered.")
        #     break

        torch.cuda.empty_cache()

    # Plot Losses
    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses, label="G")
    plt.plot(D_losses, label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig('results/Loss_During_Training_WGAN.png', pad_inches=0, bbox_inches='tight')
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.title("PSNR During Training")
    plt.plot(psnr_scores, label="PSNR")
    plt.xlabel("Epochs")
    plt.ylabel("PSNR (dB)")
    plt.legend()
    plt.savefig('results/PSNR_During_Training_WGAN.png', pad_inches=0, bbox_inches='tight')
    plt.show()


    # Save final models
    torch.save(netG.state_dict(), f'weights/final_octave_unet_generator{args.vol}.pth')
    torch.save(netD.state_dict(), f'weights/final_octave_unet_discriminator{args.vol}.pth')


def Plot_sample(test_dataloader):

    # OctaveUnet / Generator
    netG = define_G(input_nc=2, output_nc=1, norm='batch', use_dropout=False, gpu_ids=[])
    netG = netG.to(device)

    # Discriminator
    netD = define_D(input_nc=1, ndf=64, norm='batch', use_sigmoid=False, gpu_ids=[])
    netD = netD.to(device)

    optimizerG = optim.Adam(netG.parameters(), lr=1e-4, betas=(0.5, 0.999))
    optimizerD = optim.Adam(netD.parameters(), lr=1e-5, betas=(0.5, 0.999))

    # Load checkpoints if resuming training
    load_checkpoint(netG, netD, optimizerG, optimizerD)

    real_batch = next(iter(test_dataloader))

    # Plot real images
    fig = plt.figure(figsize=(24, 8))
    plt.axis("off")
    plt.title("Real Images")
    print(real_batch[2])
    real_images_grid = vutils.make_grid(real_batch[1][2:8].to(device), padding=2, normalize=True)
    plt.imshow(np.transpose(real_images_grid.cpu(), (1, 2, 0)))  # Convert channels for RGB display
    plt.savefig('results/real_images.png', pad_inches=0, bbox_inches='tight')
    plt.show()

    # Generate images using netG
    real_cpu = real_batch[0].to(device).float()
    print(real_batch[2])

    output = netG(real_cpu)

    # Handle 2-channel output, display the first channel as grayscale
    output_grid = vutils.make_grid(output[2:8, 0:1].to(device), padding=2, normalize=True)

    plt.figure(figsize=(24, 8))
    plt.axis("off")
    plt.title("Generated Images (Grayscale)")
    plt.imshow(np.transpose(output_grid.cpu(), (1, 2, 0)), cmap='gray')  # Use cmap='gray' for single channel
    plt.savefig('results/generated_images.png', pad_inches=0, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    from config import get_args
    args = get_args()
    
    if args.device == 'TPU':
        import torch_xla
        import torch_xla.core.xla_model as xm
        device = xm.xla_device()
    else:
        device = ('cuda:0' if torch.cuda.is_available() else 'cpu') 

    root = f"{args.data_dir}/"
    train_dataloader, val_dataloader = load_dataset(root+"t1_data.npy", root+"flair_data.npy", root+"t2_data.npy", root+"labels_data.npy")
    if args.plot:
      Plot_sample(val_dataloader)
    else:
      train(train_dataloader, val_dataloader)
