import argparse
import pickle
from torch_geometric.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
import os
import warnings

from model.protein_feature_generator.fusion import FAE

# Disable DeepSNAP warnings for clearer printout in the tutorial
warnings.filterwarnings("ignore")

current_file_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PATH = "{}/datasets/protein_data/model_checkpoints/Fusion_STRING.pt".format(current_file_path)
# PATH = "{}/datasets/protein_data/model_checkpoints/Fusion_Yeast.pt".format(current_file_path)


class MultimodalDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


# Load the preprocessed data
with open('{}/datasets/protein_data/fusion_STRING.pkl'.format(current_file_path), 'rb') as f:
    print("Loading data ...")
    dataset = pickle.load(f)
# with open('{}/datasets/protein_data/fusion_Yeast.pkl'.format(current_file_path), 'rb') as f:
#     print("Loading data ...")
#     dataset = pickle.load(f)
print("Data loaded successfully.")
data_shape = dataset[0].shape[0]  # 640*3

# Create a custom dataset from the loaded data
dataset = MultimodalDataset(dataset)

# Split the dataset into train, validation, and test sets
train_ratio = 0.9
valid_ratio = 0.1
test_ratio = 0.0

train_size = int(len(dataset) * train_ratio)
valid_size = int(len(dataset) * valid_ratio)
test_size = len(dataset) - train_size - valid_size

print("Train dataset size:", train_size)
print("Validation dataset size:", valid_size)
print("Test dataset size:", test_size)

# Split the dataset into train, validation, and test sets using random_split
train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(
    dataset, [train_size, valid_size, test_size]
)

batch_size = 1024

# Create data loaders for training, validation, and test
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Define loss function (Mean Squared Error) and optimizer (Adam)
criterion = nn.MSELoss()


# Training function
def train(fusion_model, train_loader, optimizer, device):
    fusion_model.train()
    total_loss = 0.0

    for batch in train_loader:
        optimizer.zero_grad()
        batch = batch.to(device)
        encoding = fusion_model.encode(batch)
        restoration = fusion_model.decode(encoding)
        loss = criterion(batch, restoration)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    average_loss = total_loss / len(train_loader)
    return average_loss


# Validation function
def validation(fusion_model, valid_loader, device):
    fusion_model.eval()
    total_val_loss = 0.0

    with torch.no_grad():
        for batch in valid_loader:
            batch = batch.to(device)
            encoding = fusion_model.encode(batch)
            restoration = fusion_model.decode(encoding)
            val_loss = criterion(batch, restoration)
            total_val_loss += val_loss.item()

    average_val_loss = total_val_loss / len(valid_loader)
    return average_val_loss


# Test function
def test_model(fusion_model, test_loader, device):
    fusion_model.eval()
    total_test_loss = 0.0

    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            encoding = fusion_model.encode(data)
            restoration = fusion_model.decode(encoding)
            test_loss = criterion(data, restoration)
            total_test_loss += test_loss.item()

    average_test_loss = total_test_loss / len(test_loader)
    return average_test_loss


def main():
    parser = argparse.ArgumentParser(description="FAE")
    parser.add_argument("--mode", choices=["train", "test"], help="Select mode: train or test", required=False, default="train")
    args = parser.parse_args()

    input_features = data_shape
    latent_dim = 1024
    fusion_model = FAE(latent_dim, input_features)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    optimizer = optim.Adam(fusion_model.parameters(), lr=0.001)
    fusion_model = fusion_model.to(device)

    if args.mode == "train":
        # Training mode
        num_epochs = 200
        for epoch in range(num_epochs):
            train_loss = train(fusion_model, train_loader, optimizer, device)

            val_loss = validation(fusion_model, valid_loader, device)

            print(f"Epoch [{epoch + 1}/{num_epochs}] Train Loss: {train_loss:.4f} - Validation Loss: {val_loss:.4f}")

        # Save the fusion model to a file
        torch.save(fusion_model, PATH)
        print("Model saved:{}".format(PATH))

    elif args.mode == "test":
        # Test mode
        # Load the saved model
        fusion_model = torch.load(PATH)
        # Evaluate the model on the test dataset
        test_loss = test_model(fusion_model, test_loader, device)
        print(f"Average MSE on Test Set: {test_loss:.4f}")


if __name__ == "__main__":
    main()
