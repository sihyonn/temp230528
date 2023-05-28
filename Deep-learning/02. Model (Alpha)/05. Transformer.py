import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

class TransformerModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(TransformerModel, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size

        self.encoder = nn.Linear(input_size, hidden_size)
        self.transformer = nn.Transformer(hidden_size, num_layers)
        self.decoder = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.encoder(x)
        x = self.transformer(x)
        x = self.decoder(x)
        return x

class DamDataset(Dataset):
    def __init__(self, excel_file):
        # Load the data
        self.data = pd.read_excel(excel_file)

        # Normalize the data (except year and month columns)
        for column in self.data.columns[2:]:
            self.data[column] = (self.data[column] - self.data[column].mean()) / self.data[column].std()

        # Extract the labels
        self.labels = self.data['Dam_Level'].values

        # Drop the labels from the input data
        self.data = self.data.drop('Dam_Level', axis=1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Convert to tensors
        sample = torch.Tensor(self.data.iloc[idx].values)
        label = torch.Tensor([self.labels[idx]])

        return sample, label

def get_loader(excel_file, batch_size):
    dataset = DamDataset(excel_file)
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

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    model = TransformerModel(input_size, hidden_size, num_layers, output_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_loader = get_loader('your_excel_file.xlsx', batch_size)
    loss_months = [0]*12

    for epoch in range(1, epochs + 1):
        train(model, device, train_loader, optimizer, epoch, loss_months)

if __name__ == '__main__':
    main()