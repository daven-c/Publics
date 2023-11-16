from TorchModels import Generator, Discriminator
import os
import torch
from torch import nn
import torchvision
import torchvision.transforms as transforms
from ViewResults import *
from datetime import datetime
import os
import shutil
import time
import math


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"{device} enabled")

    # Load configs
    configs = load_configs()
    print(configs)

    # Options
    num_classes = 10
    noise_amount = configs["noise_amount"]
    epochs = configs["epochs"]
    save_interval = configs["save_interval"]
    batch_size = configs["batch_size"]
    learning_rate = configs["learning_rate"]
    beta1 = configs["beta_1"]
    skip_preview = configs["skip_preview"]

    # Load the MNIST dataset.
    transform = transforms.Compose(
        [transforms.ToTensor()])  # transforms.Normalize((0.5,), (0.5,))
    dataset = torchvision.datasets.MNIST(
        root='./data', train=True, transform=transform, download=True)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True)

    # Model instantiation
    generator = Generator(input_size=noise_amount +
                          num_classes, hidden_dim=256, output_size=28 * 28, lr=learning_rate, beta1=beta1)
    discriminator = Discriminator(
        input_size=noise_amount + num_classes, hidden_dim=256, output_size=1, lr=learning_rate, beta1=beta1)

    gen_criterion = nn.CrossEntropyLoss()
    disc_criterion = nn.BCELoss()
    gan_criterion = nn.BCELoss()

    # Create save directory
    current_datetime = datetime.now()
    formatted_datetime = current_datetime.strftime("%m-%d-%y_%H.%M")
    folder_name = f"GAN_{formatted_datetime}"
    folder_path = "./models/" + folder_name
    os.makedirs(folder_path, exist_ok=True)
    os.makedirs(folder_path + "/images", exist_ok=True)
    shutil.copy("settings.cfg", folder_path)
    print("Directory created at " + folder_path)

    # Train the GAN.
    print(f"{'Epoch':<20} {'Batch':^20} {'Time':^20} {'Disc Loss':^20} {'GAN Loss':^20}")
    num_batches = math.ceil(len(dataloader.dataset) / batch_size)
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        print(
            f"{f'{epoch}/{epochs}':<20} {f'{1}/{num_batches}':^20} {f'{0}s':^20} {'0':^20} {f'0':^20}", end='\r')
        for batch, (images, labels) in enumerate(dataloader):
            curr_batch_size = images.shape[0]

            # Data generation and preprocess
            rdn_noise = torch.rand(
                size=(curr_batch_size, noise_amount))
            batch_images = images.view(images.shape[0], -1)  # Flatten images
            class_labels = multi_one_hot_label(
                labels)  # hot encodes class labels
            real_labels = torch.ones(curr_batch_size)
            fake_labels = torch.zeros(curr_batch_size)

            # Train the generator
            generator.optimizer.zero_grad()
            fake_images = generator(rdn_noise, class_labels)
            gen_loss = gen_criterion(fake_images, batch_images)
            gen_loss.backward()
            generator.optimizer.step()

            # Train the discriminator
            discriminator.optimizer.zero_grad()
            d_loss_real = disc_criterion(
                discriminator(batch_images, class_labels), real_labels)
            fake_images = generator(rdn_noise, class_labels)
            d_loss_fake = disc_criterion(discriminator(
                fake_images, class_labels), fake_labels)
            d_loss = (d_loss_real + d_loss_fake) / 2
            d_loss.backward()
            discriminator.optimizer.step()

            # Train the gan
            generator.optimizer.zero_grad()
            real_labels = torch.ones(curr_batch_size)
            fake_images = generator(rdn_noise, class_labels)
            gan_loss = gan_criterion(discriminator(
                fake_images, class_labels), real_labels)
            gan_loss.backward()
            generator.optimizer.step()
            print(
                f"{f'{epoch}/{epochs}':<20} {f'{batch + 1}/{num_batches}':^20} {f'{(time.time() - start_time):.2f}s':^20} {f'{d_loss:.4f}':^20} {f'{gan_loss:.4f}':^20}", end='\r')
        print()
        if (epoch % save_interval == 0) or (epoch == epochs + 1):
            plot = preview_all(
                generator, noise_amount=noise_amount, return_only=skip_preview)
            plot.savefig(f"{folder_path}/images/epoch_{epoch}")
            torch.save(generator.state_dict(),
                       f"{folder_path}/GenModelE{epoch}.pth")
            # print(f"model saved as: {folder_path}/GenModelE{epoch}.pth")

    print("training finished")
