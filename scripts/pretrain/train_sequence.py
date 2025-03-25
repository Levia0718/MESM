import argparse
import warnings
import random
import pickle
import torch
from torch.utils.data import Dataset, DataLoader
import os

from model.protein_feature_generator.sequence import SVAE

random.seed(0)

# Disable DeepSNAP warnings for clearer printout in the tutorial
warnings.filterwarnings("ignore")

current_file_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PATH = "{}/datasets/protein_data/model_checkpoints/Seq_VAE_STRING.pt".format(current_file_path)
# PATH = "{}/datasets/protein_data/model_checkpoints/Seq_VAE_Yeast.pt".format(current_file_path)


with open('{}/datasets/protein_data/sequences_STRING.pkl'.format(current_file_path), 'rb') as f:
    print("Loading data ...")
    protein_vecs = pickle.load(f)
# with open('{}/datasets/protein_data/sequences_Yeast.pkl'.format(current_file_path), 'rb') as f:
#     print("Loading data ...")
#     protein_vecs = pickle.load(f)
print("Data loaded successfully.")


class SequenceDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def split_multi_protein_vecs(data, train_ratio=0.9, test_ratio=0.0, valid_ratio=0.1):
    total_data = len(data)
    indices = list(range(total_data))
    random.shuffle(indices)

    num_train = int(total_data * train_ratio)
    num_test = int(total_data * test_ratio)
    num_valid = int(total_data * valid_ratio)

    print("Train dataset size:", num_train)
    print("Validation dataset size:", num_valid)
    print("Test dataset size:", num_test)

    train_indices = indices[:num_train]
    test_indices = indices[num_train:num_train + num_test]
    valid_indices = indices[num_train + num_test:]

    train_data = data[train_indices]
    test_data = data[test_indices]
    valid_data = data[valid_indices]

    return train_data, test_data, valid_data


train_data, test_data, valid_data = split_multi_protein_vecs(protein_vecs)

sequence_dataset = SequenceDataset(protein_vecs)

train_dataset = SequenceDataset(train_data)
test_dataset = SequenceDataset(test_data)
valid_dataset = SequenceDataset(valid_data)

batch_size = 256

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)


def train(model, train_loader, optimizer, device):
    model.train()
    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        x_reconstructed, mean, log_var, encoder_output = model(data)
        loss = model.compute_loss(data, x_reconstructed, mean, log_var)
        loss.backward()
        optimizer.step()
        total_loss += loss
    return total_loss / len(train_loader)


def validation(model, valid_loader, device):
    model.eval()
    with torch.no_grad():
        val_loss = 0
        for data in valid_loader:
            data = data.to(device)
            x_reconstructed, mean, log_var, encoder_output = model(data)
            loss = model.compute_loss(data, x_reconstructed, mean, log_var)
            val_loss += loss.item()
        val_loss /= len(valid_loader)
        return val_loss


def test_model(model, test_loader, device):
    model.eval()
    AUC = []
    AP = []

    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            z = model.encode(data.x, data.edge_index)
            auc, ap = model.test(z, data.edge_index, data.neg_edge_index)
            AUC.append(auc)
            AP.append(ap)
    return sum(AUC) / len(AUC), sum(AP) / len(AP)


def main():
    parser = argparse.ArgumentParser(description="SVAE")
    parser.add_argument("--mode", choices=["train", "test"], help="Select mode: train or test", required=False, default="train")
    args = parser.parse_args()

    vae_model = SVAE(input_length=2000, input_dim=13, latent_dim=1024)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vae_model = vae_model.to(device)

    optimizer = torch.optim.Adam(vae_model.parameters(), lr=0.001)

    num_epochs = 200
    if args.mode == "train":
        for epoch in range(1, num_epochs + 1):
            train_loss = train(vae_model, train_loader, optimizer, device)
            val_loss = validation(vae_model, valid_loader, device)
            print(f'Epoch [{epoch}/{num_epochs}], Train Loss: {train_loss:.4f}, Valid Loss: {val_loss:.4f}')

        torch.save(vae_model, PATH)
        print("Model saved:{}".format(PATH))

    elif args.mode == "test":
        vae_model = torch.load(PATH)

        AUC, AP = test_model(vae_model, test_loader, device)
        print(f"AUC: {AUC}, AP: {AP}")


if __name__ == "__main__":
    main()
