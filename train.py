import torch
import torch.nn as nn
import numpy as np
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch import optim
from torch.utils.data import DataLoader
from PIL import Image
import os
import json
from model import Discriminator, Generator, initialize_weights


'''# Loads model from path
def load_model(model, path):
    model.load_state_dict(torch.load(path))
    model.eval()
    return model


# Generates image from saved model
def generate_image(model, path):
    model = load_model(model, path)
    noise = torch.randn((1, NOISE_DIM, 1, 1)).to(device)
    fake = model(noise)
    fake = fake.reshape(3, IMAGE_SIZE, IMAGE_SIZE)
    fake = fake.detach().numpy()
    fake = np.transpose(fake, (1, 2, 0))
    fake = (fake + 1) / 2
    fake = fake * 255
    fake = fake.astype(np.uint8)
    fake = Image.fromarray(fake)
    fake.show()'''


# Convert the images from palette to RGBA
class palletteToRGBA(object):
    def __call__(self, img):
        if img.mode == "P":
            img = img.convert("RGBA")
        return img


# Define Hyperparameters
device = torch.device("cpu")
LEARNING_RATE = .01
BATCH_SIZE = 128
IMAGE_SIZE = 64
CHANNELS_IMG = 3
NOISE_DIM = 100
NUM_EPOCHS = 10
FEATURES_DISK = 64
FEATURES_GEN = 64

# Define the transformations to be applied to the images
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),    # resize the images to 64x64
    palletteToRGBA(),                             # convert the images from palette to RGBA
    transforms.ToTensor(),                          # convert the images to PyTorch tensors
    transforms.Normalize(mean=[0.5], std=[0.5])     # normalize the pixel values to [-1, 1]
])

# Gets dataset from the folder
dataset = datasets.ImageFolder(root="Humans", transform=transform)
print("Successfully loaded dataset")
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
gen = Generator(NOISE_DIM, CHANNELS_IMG, FEATURES_GEN).to(device)
disc = Discriminator(CHANNELS_IMG, FEATURES_DISK).to(device)
initialize_weights(gen)
initialize_weights(disc)

with open("epoch_num.json", "r") as f:
    data = json.load(f)
    prev_epoch_num = data["epoch_num"]

if prev_epoch_num != 0:
    gen.load_state_dict(torch.load("generator.pth"))
    print("Generator loaded")
    disc.load_state_dict(torch.load("discriminator.pth"))
    print("Discriminator loaded")


opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.5, .999))
opt_disc = optim.Adam(disc.parameters(), lr=LEARNING_RATE, betas=(0.5, .999))
criterion = nn.BCELoss()

fixed_noise = torch.randn(32, NOISE_DIM, 1, 1).to(device)
step = 0

gen.train()
disc.train()
print("Starting Training")

for epoch in range(NUM_EPOCHS):
    for batch_idx, (real, _) in enumerate(loader):
        real = real.to(device)
        noise = torch.randn((BATCH_SIZE, NOISE_DIM, 1, 1)).to(device)
        fake = gen(noise)

        # Train Discriminator
        disc_real = disc(real).reshape(-1)  # N
        loss_disc_real = criterion(disc_real, torch.ones_like(disc_real))
        disc_fake = disc(fake).reshape(-1)
        loss_disc_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
        loss_disc = (loss_disc_real + loss_disc_fake) / 2
        disc.zero_grad()
        loss_disc.backward(retain_graph=True)
        opt_disc.step()

        # Train Generator
        output = disc(fake).reshape(-1)
        loss_gen = criterion(output, torch.ones_like(output))
        gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()

        # Print losses
        print(
            f"Epoch [{prev_epoch_num+epoch+1}/{prev_epoch_num+NUM_EPOCHS}] Batch {batch_idx+1}/{len(loader)} \
                            Loss D: {loss_disc:.4f}, loss G: {loss_gen:.4f}"
        )

        if batch_idx % 50 == 0:
            with torch.no_grad():
                fake = gen(fixed_noise)
                # take out (up to) 32 examples
                img_grid_real = torchvision.utils.make_grid(
                    real[:32], normalize=True
                )

                img_grid_fake = torchvision.utils.make_grid(
                    fake[:32], normalize=True
                )

                # If the directory "output" does not exist, create it
                try:
                    os.mkdir("output")
                except FileExistsError:
                    pass

                # save images to file directory
                if batch_idx == 0:
                    torchvision.utils.save_image(img_grid_real, f'output/real_images_epoch{prev_epoch_num+epoch+1}.png')
                torchvision.utils.save_image(img_grid_fake, f'output/fake_images_epoch{prev_epoch_num+epoch+1}'
                                                            f'_batch{batch_idx}.png')
                print("Image for epoch {} and batch {} saved to file directory".format(prev_epoch_num+epoch+1,
                                                                                       batch_idx))

            step += 1

        if epoch == NUM_EPOCHS - 1 and batch_idx == len(loader) - 1:
            torchvision.utils.save_image(img_grid_fake, f'output/final_image.png')
            print("Final image saved to file directory")

# Open and update number of epochs ran in epoch_num.json
with open("epoch_num.json", "r") as f:
    data = json.load(f)
    data["epoch_num"] += NUM_EPOCHS
with open("epoch_num.json", "w") as f:
    json.dump(data, f)

# Replace models
if os.path.exists("generator.pth"):
    os.remove("generator.pth")
if os.path.exists("discriminator.pth"):
    os.remove("discriminator.pth")
torch.save(gen.state_dict(), "generator.pth")
torch.save(disc.state_dict(), "discriminator.pth")
print("Models saved to file directory")

