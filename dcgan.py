import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
from torchvision.transforms import transforms
from torchvision.utils import save_image

parser = argparse.ArgumentParser()
parser.add_argument('--device', required=False, default='cuda')
parser.add_argument('--nz', required=False, default=100)
parser.add_argument('--ngf', required=False, default=64)
parser.add_argument('--ndf', required=False, default=64)
parser.add_argument('--lr', required=False, default=0.0002)
parser.add_argument('--beta1', required=False, default=0.5)
parser.add_argument('--batch_size', required=False, default=128)
parser.add_argument('--epochs', required=True, help='training epochs')
args = parser.parse_args()


class Generator(nn.Module):
    def __init__(self, nz, img_channels):
        super(Generator, self).__init__()

        self.in_channels = nz
        self.out_channels = img_channels

        self.conv = nn.Sequential(
            nn.ConvTranspose2d(self.in_channels, ngf * 8, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf, self.out_channels, 4, 2, 1, bias=False)
        )

    def forward(self, x):
        x = self.conv(x)
        x = torch.tanh(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, img_channels):
        super(Discriminator, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(img_channels, ndf, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, 3, 2, 1),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, ndf * 4, 3, 2, 1),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 4, ndf * 8, 3, 2, 1),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False)
        )

    def forward(self, x):
        x = self.conv(x)
        x = torch.sigmoid(x)
        return x


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


def train_dcgan(epochs, save_interval):
    generator.train()
    discriminator.train()
    for epoch in range(epochs):
        for n, (img, _) in enumerate(dataloader):
            img = img.to(device)

            # Generator
            optimizer_g.zero_grad()

            z = torch.randn(batch_size, nz, 1, 1).to(device)
            gen_out = generator(z)
            output = discriminator(gen_out)
            score_dgz = output.mean().item()

            loss_g = loss_fn(output, torch.ones_like(output))
            loss_g.backward()
            optimizer_g.step()

            # Discriminator
            optimizer_d.zero_grad()

            output = discriminator(img)
            score_dx = output.mean().item()

            loss_d_real = loss_fn(output, torch.ones_like(output))

            output = discriminator(gen_out.detach())
            loss_d_fake = loss_fn(output, torch.zeros_like(output))

            loss_d = (loss_d_fake + loss_d_real) / 2
            loss_d.backward()
            optimizer_d.step()

            if (n + 1) % save_interval == 0:
                print('epoch: [{}/{}], step: [{}/{}], loss_g: {:.4f}, loss_d: {:.4f}, D(G(z): {:.4f}, D(x): {:.4f}'
                      .format(epoch + 1, epochs, n + 1, len(dataloader), loss_g.item(), loss_d.item(), score_dgz,
                              score_dx))
                # random_imgs = gen_out[torch.randperm(gen_out.shape[0])[:25]]
                # save_image(random_imgs, f'dcgan_img/{dataset_name}/{epoch + 1}_{n + 1}.png', nrow=5, normalize=True)


if __name__ == '__main__':
    device = torch.device('cuda' if args.device == 'cuda' and torch.cuda.is_available() else 'cpu')
    nz = args.nz
    ngf = args.ngf
    ndf = args.ndf
    lr = args.lr
    beta1 = args.beta1
    batch_size = args.batch_size
    epochs = int(args.epochs)
    dataset = datasets.CIFAR10
    dataset_name = dataset.__module__.split('.')[-1]
    img_channels = 1 if dataset_name == 'mnist' else 3

    transform = transforms.Compose([transforms.Resize((64, 64)),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.5] * img_channels,
                                                         [0.5] * img_channels)])

    train_dataset = dataset('../dataset/', train=True, download=True, transform=transform)
    test_dataset = dataset('../dataset/', train=False, download=True, transform=transform)
    dataset = train_dataset + test_dataset
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    generator = Generator(nz, img_channels).to(device)
    discriminator = Discriminator(img_channels).to(device)

    optimizer_g = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, 0.999))
    loss_fn = nn.BCELoss()

    generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)

    train_dcgan(epochs=epochs, save_interval=len(dataloader) // 3)