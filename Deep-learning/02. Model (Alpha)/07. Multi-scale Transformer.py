import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

class MultiScaleTransformerModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, num_scales):
        super(MultiScaleTransformerModel, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.num_scales = num_scales

        self.encoders = nn.ModuleList([nn.Linear(input_size, hidden_size) for _ in range(num_scales)])
        self.transformers = nn.ModuleList([nn.Transformer(hidden_size, num_layers) for _ in range(num_scales)])
        self.decoder = nn.Linear(hidden_size * num_scales, output_size)

    def forward(self, x):
        outs = []
        for i in range(self.num_scales):
            encoded = self.encoders[i](x[:, i, :, :])
            transformed = self.transformers[i](encoded)
            outs.append(transformed)
        concatenated = torch.cat(outs, dim=-1)
        decoded = self.decoder(concatenated)
        return decoded


class DamDataset(Dataset):
    def __init__(self, excel_file, scales):
        # Initialize and preprocess the data
        ...
        self.scales = scales

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Prepare the samples for each scale
        samples = []
        for scale in self.scales:
            start = max(0, idx - scale + 1)
            sample = self.data.iloc[start:idx + 1].values
            samples.append(sample)
        samples = torch.Tensor(samples)

        label = torch.Tensor([self.labels[idx]])

        return samples, label


def get_loader(excel_file, scales, batch_size):
    dataset = DamDataset(excel_file, scales)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader

def train(model, device, train_loader, optimizer, epoch, loss_months):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)

        # Compute the loss for the current month
        loss = nn.MSELoss()(output, target)
        loss_months[epoch % 12] = loss.item()

        # Compute the average loss over the last 12 months
        avg_loss = sum(loss_months) / len(loss_months)
        print(f'Epoch: {epoch}, Loss: {avg_loss}')

        loss.backward()
        optimizer.step()

def main():
    # Training settings
    input_size = 6  # input dimension
    hidden_size = 32  # hidden layer dimension
    num_layers = 2  # number of hidden layers
    output_size = 1  # output dimension
    learning_rate = 0.001  # initial learning rate
    epochs = 1000  # number of epochs
    batch_size = 64  # batch size
    scales = [12, 24, 36, 48]  # list of past months to consider

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    model = MultiScaleTransformerModel(input_size, hidden_size, output_size, num_layers, len(scales)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_loader = get_loader('your_excel_file.xlsx', scales, batch_size)
    loss_months = [0]*12

    for epoch in range(1, epochs + 1):
        train(model, device, train_loader, optimizer, epoch, loss_months)

if __name__ == '__main__':
    main()