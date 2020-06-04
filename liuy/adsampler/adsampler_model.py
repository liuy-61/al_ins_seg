import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)


class VAE(nn.Module):
    """Encoder-Decoder architecture for both WAE-MMD and WAE-GAN."""
    def __init__(self, image_size=32, z_dim=32, nc=3):
        super(VAE, self).__init__()
        self.z_dim = z_dim
        self.nc = nc
        self.encoder = nn.Sequential(
            nn.Conv2d(nc, 128, 4, 2, 1, bias=False),             # input: bs, is, is  output: 128, is/2, is/2
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),            # input: 128, is/2, is/2  output: 256, is/4, is/4
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),            # input: 256, is/4, is/4  output: 512, is/8, is/8
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(512, 1024, 4, 2, 1, bias=False),           # input: 512, is/8, is/8  output: 1024, is/16, is/16
            nn.BatchNorm2d(1024),
            nn.ReLU(True),
            View((-1, int(1024 * image_size/16 * image_size/16))),                                # input: 1024,is/16, is/16  output: bs,1024*is/16*is/16
        )

        self.fc_mu = nn.Linear(1024 * int((image_size/16)) * int((image_size/16)), z_dim)                   # input: bs,1024*is/16*is/16  output: bs,z_dim
        self.fc_logvar = nn.Linear(1024 * int((image_size/16)) * int((image_size/16)), z_dim)               # input: bs,1024*is/16*is/16  output: bs,z_dim
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, 1024 * int((image_size/8)) * int((image_size/8))),                           # input: 1024, is/16 , is/16  output: 1024, is/8, is/8
            View((-1, 1024, int((image_size/8)), int((image_size/8)))),                               # input: 1024, is/8, is/8  output: 1024, is/8, is/8
            nn.ConvTranspose2d(1024, 512, 4, 2, 1, bias=False),   # input: 1024, is/8, is/8  output: 512, is/4, is/4
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),    # input: 512, is/4, is/4  output: 256, is/2, is/2
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),    # input: 256, is/2, is/2  output: 128, is, is
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, nc, 1),                       # input: 128, is, is  output: nc, is, is
        )
        self.weight_init()

    def weight_init(self):
        for block in self._modules:
            try:
                for m in self._modules[block]:
                    kaiming_init(m)
            except:
                kaiming_init(block)

    def forward(self, x):
        z = self._encode(x)
        mu, logvar = self.fc_mu(z), self.fc_logvar(z)
        z = self.reparameterize(mu, logvar)
        x_recon = self._decode(z)

        return x_recon, z, mu, logvar

    def reparameterize(self, mu, logvar):
        stds = (0.5 * logvar).exp()
        epsilon = torch.randn(*mu.size())
        if mu.is_cuda:
            stds, epsilon = stds.cuda(), epsilon.cuda()
        latents = epsilon * stds + mu
        return latents

    def _encode(self, x):
        return self.encoder(x)

    def _decode(self, z):
        return self.decoder(z)


class Discriminator(nn.Module):
    """Adversary architecture(Discriminator) for WAE-GAN."""
    def __init__(self, z_dim=32):
        super(Discriminator, self).__init__()
        self.z_dim = z_dim
        self.net = nn.Sequential(
            nn.Linear(z_dim, 512),
            nn.ReLU(True),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )
        self.weight_init()

    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)

    def forward(self, z):
           return self.net(z)


def kaiming_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.kaiming_normal(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)


def normal_init(m, mean, std):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        m.weight.data.normal_(mean, std)
        if m.bias.data is not None:
            m.bias.data.zero_()
    elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
        m.weight.data.fill_(1)
        if m.bias.data is not None:
            m.bias.data.zero_()
