import argparse
import warnings
import random
import pickle
from torch_geometric.data import DataLoader
import os

from model.protein_feature_generator.structure import *

random.seed(0)

# Disable DeepSNAP warnings for clearer printout in the tutorial
warnings.filterwarnings("ignore")

# Define the file path for saving the model
current_file_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PATH = "{}/datasets/protein_data/model_checkpoints/VGAE_STRING.pt".format(current_file_path)
# PATH = "{}/datasets/protein_data/model_checkpoints/VGAE_Yeast.pt".format(current_file_path)


# Load graph data from a pickle file
with open('{}/datasets/protein_data/graphs_STRING.pkl'.format(current_file_path), 'rb') as f:
    print("Loading data ...")
    graphs = pickle.load(f)
# with open('{}/datasets/protein_data/graphs_Yeast.pkl'.format(current_file_path), 'rb') as f:
#     print("Loading data ...")
#     graphs = pickle.load(f)
print("Data loaded successfully.")


# Function to split multi graphs into train, test, and validation sets
def split_multi_graphs(graphs, train_ratio=0.9, test_ratio=0.0, valid_ratio=0.1):
    # Shuffle the list of graphs to ensure randomness
    random.shuffle(graphs)

    # Calculate the number of graphs for each split
    total_graphs = len(graphs)
    num_train = int(total_graphs * train_ratio)
    num_test = int(total_graphs * test_ratio)
    num_valid = int(total_graphs * valid_ratio)
    print("Train dataset size:", num_train)
    print("Validation dataset size:", num_valid)
    print("Test dataset size:", num_test)

    # Split the dataset
    train_graphs = graphs[:num_train]
    test_graphs = graphs[num_train:num_train + num_test]
    valid_graphs = graphs[num_train + num_test:]

    return train_graphs, test_graphs, valid_graphs


# Split the graphs into train, test, and validation sets
train_graphs, test_graphs, valid_graphs = split_multi_graphs(graphs)

batch_size = 256

# Create data loaders for train, test, and validation sets
train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_graphs, batch_size=batch_size, shuffle=False)
valid_loader = DataLoader(valid_graphs, batch_size=batch_size, shuffle=False)


# Function to train the VGAE model
def train(model, train_loader, optimizer, device):
    model.train()
    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        z = model.encode(data.x, data.edge_index)
        loss = model.recon_loss(z, data.edge_index, data.neg_edge_index)
        loss.backward()
        optimizer.step()
        total_loss += loss
    return total_loss / len(train_loader)


# Function to perform validation on the VGAE model
def validation(model, valid_loader, device):
    model.eval()
    with torch.no_grad():
        val_loss = 0
        for data in valid_loader:
            data = data.to(device)
            z = model.encode(data.x, data.edge_index)
            loss = model.recon_loss(z, data.edge_index, data.neg_edge_index)
            val_loss += loss.item()
        val_loss /= len(valid_loader)
        return val_loss


# Function to test the VGAE model
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
    parser = argparse.ArgumentParser(description="VGAE")
    parser.add_argument("--mode", choices=["train", "test"], help="Select mode: train or test", required=False, default="train")
    args = parser.parse_args()

    # Define the output dimensions for the model
    out_channels = 10
    num_features = 21

    # Create an instance of VGAE using the VariationalGCNEncoder
    vgae_model = VGAE(VariationalGCNEncoder(num_features, out_channels))

    # Check if GPU is available, and move the model to the appropriate device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vgae_model = vgae_model.to(device)

    # Create an optimizer for the VGAE (Variational Graph Autoencoder) model
    optimizer = torch.optim.Adam(vgae_model.parameters(), lr=0.001)

    # Your existing code here
    num_epochs = 200
    if args.mode == "train":
        # Training mode
        for epoch in range(1, num_epochs + 1):
            train_loss = train(vgae_model, train_loader, optimizer, device)
            val_loss = validation(vgae_model, valid_loader, device)
            print(f'Epoch [{epoch}/{num_epochs}], Train Loss: {train_loss:.4f}, Valid Loss: {val_loss:.4f}')

        # Save the VGAE model
        torch.save(vgae_model, PATH)
        print("Model saved:{}".format(PATH))

    elif args.mode == "test":
        # Test mode
        # Load the saved model
        vgae_model = torch.load(PATH)

        # Test the model on the test dataset
        AUC, AP = test_model(vgae_model, test_loader, device)
        print(f"AUC: {AUC}, AP: {AP}")


if __name__ == "__main__":
    main()
