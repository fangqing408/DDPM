import torch
from torch import nn
from models.unet import DDPMUNet
from models.ddpm import DDPM

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from tqdm.auto import tqdm

def train(epochs: int, dataloader: DataLoader, ddpm: DDPM, device: torch.device):
    ddpm.train()
    optimizer = torch.optim.Adam(ddpm.parameters(), lr=1e-4)
    loss_fn = nn.MSELoss()
    for epoch in range(epochs):
        pbar = tqdm(dataloader, total=len(dataloader), desc=f'Epoch {epoch+1}/{epochs}')
        total_loss = 0.0
        total_samples = 0.0
        for x_0, _ in pbar:
            x_0: torch.Tensor = x_0.to(device)
            t = torch.randint(0, ddpm.timesteps, (x_0.shape[0], ), device=device)
            eps = torch.randn_like(x_0, device=device)
            x_t = ddpm.q_sample(x_0, t, eps)
            eps_theta = ddpm.eps_model(x_t, t)
            loss: torch.Tensor = loss_fn(eps_theta, eps)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * x_0.shape[0]
            total_samples += x_0.shape[0]

            pbar.set_postfix({'avg_loss': f"{total_loss / total_samples:.4f}", "batch_loss": f"{loss.item() / x_0.shape[0]:.4f}"})
        print(f'Epoch {epoch+1}/{epochs} finished. average Loss: {total_loss / total_samples:.4f}')
        

if __name__ == '__main__':
    dataset = datasets.MNIST('./data', download=True, transform=transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor()
    ]))
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ddpm_unet = DDPMUNet(1, 128, num_res_blocks=3)
    ddpm = DDPM(1000, ddpm_unet).to(device)

    train(5, dataloader, ddpm, device)

    torch.save(ddpm.state_dict(), './model_ckpts/ddpm.pth')

