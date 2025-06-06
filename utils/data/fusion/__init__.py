import pickle
import os
from torch_geometric.nn import TopKPooling

from model.protein_feature_generator.sequence import SVAE
from model.protein_feature_generator.structure import *
from model.protein_feature_generator.pointcloud import *
from model.protein_feature_generator.fusion import *

# Check if GPU is available
if torch.cuda.is_available():
    device = torch.device('cuda')
    print("Using GPU:", torch.cuda.get_device_name(0))
else:
    device = torch.device('cpu')
    print("No GPU available, using CPU.")

current_file_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Load pre-trained models
script_directory = os.path.dirname(os.path.abspath(__file__))
vae_model = torch.load("{}/datasets/protein_data/model_checkpoints/Seq_VAE_STRING.pt".format(current_file_path), map_location=device)
vgae_model = torch.load("{}/datasets/protein_data/model_checkpoints/VGAE_STRING.pt".format(current_file_path), map_location=device)
pae_model = torch.load("{}/datasets/protein_data/model_checkpoints/PAE_STRING.pt".format(current_file_path), map_location=device)
# vae_model = torch.load("{}/datasets/protein_data/model_checkpoints/Seq_VAE_Yeast.pt".format(current_file_path), map_location=device)
# vgae_model = torch.load("{}/datasets/protein_data/model_checkpoints/VGAE_Yeast.pt".format(current_file_path), map_location=device)
# pae_model = torch.load("{}/datasets/protein_data/model_checkpoints/PAE_Yeast.pt".format(current_file_path), map_location=device)
print("Pre-trained models loaded successfully.")

data_folder = "{}/datasets/protein_data".format(current_file_path)

with open('{}/sequences_STRING.pkl'.format(data_folder), 'rb') as f:
    print("Loading sequence data ...")
    sequence_data = pickle.load(f)
print("Sequence data loaded successfully.")
# with open('{}/sequences_Yeast.pkl'.format(data_folder), 'rb') as f:
#     print("Loading sequence data ...")
#     sequence_data = pickle.load(f)
# print("Sequence data loaded successfully.")

with open('{}/graphs_STRING.pkl'.format(data_folder), 'rb') as f:
    print("Loading graph data ...")
    graph_data = pickle.load(f)
print("Graph data loaded successfully.")
# with open('{}/graphs_Yeast.pkl'.format(data_folder), 'rb') as f:
#     print("Loading graph data ...")
#     graph_data = pickle.load(f)
# print("Graph data loaded successfully.")

with open('{}/pointclouds_STRING.pkl'.format(data_folder), 'rb') as f:
    print("Loading point cloud data ...")
    point_cloud_data = pickle.load(f)
print("Point Cloud data loaded successfully.")
# with open('{}/pointclouds_Yeast.pkl'.format(data_folder), 'rb') as f:
#     print("Loading point cloud data ...")
#     point_cloud_data = pickle.load(f)
# print("Point Cloud data loaded successfully.")


# Function for Z-score standardization
def z_score_standardization(tensor):
    mean = tensor.mean()
    std = tensor.std()
    if std != 0:
        standardized_tensor = (tensor - mean) / std
    else:
        standardized_tensor = tensor  # Handle the case when std is 0
    return standardized_tensor


# Function to process and adjust encoded graph data to a fixed size
def process_encoded_graph(encoded_graph, edge_index, fixed_size=640, feature_dim=10):
    num_nodes = encoded_graph.size(0)
    if num_nodes > fixed_size:
        ratio = fixed_size / num_nodes
        with torch.no_grad():
            pooling_layer = TopKPooling(in_channels=feature_dim, ratio=ratio)
            pooled_x, edge_index, edge_attr, batch, perm, score = pooling_layer(encoded_graph, edge_index)

        processed_encoded_graph = pooled_x[:fixed_size, :]
    else:
        padding_size = fixed_size - num_nodes
        zero_padding = torch.zeros(padding_size, feature_dim)
        processed_encoded_graph = torch.cat((encoded_graph, zero_padding), dim=0)
    return processed_encoded_graph


processed_data_list = []

for i, (graph, point_cloud, sequence) in enumerate(zip(graph_data, point_cloud_data, sequence_data)):
    # Encode sequence data using SVAE
    with torch.no_grad():
        sequence = sequence.to(device)
        sequence = sequence.unsqueeze(0)
        encoded_sequence = vae_model.encoder.get_encoder_output(sequence).to("cpu")

        encoded_sequence = z_score_standardization(encoded_sequence)

    # Encode graph data using VGAE
    with torch.no_grad():
        graph = graph.to(device)
        encoded_graph = vgae_model.encode(graph.x, graph.edge_index).to("cpu")
        graph = graph.to("cpu")
        encoded_graph = process_encoded_graph(encoded_graph, graph.edge_index)
        encoded_graph = torch.mean(encoded_graph, dim=1)
        encoded_graph = z_score_standardization(encoded_graph)

    # Encode point cloud data using PAE
    with torch.no_grad():
        point_cloud = point_cloud.to(device)
        encoded_point_cloud = pae_model.encode(point_cloud[None, :]).squeeze().to("cpu")
        encoded_point_cloud = z_score_standardization(encoded_point_cloud)

    concatenated_data = torch.cat((encoded_sequence, encoded_graph, encoded_point_cloud), dim=0)
    processed_data_list.append(concatenated_data)

print("Done")

# Print the shapes of the encoded data
print("Encoded Sequence Shape:", encoded_sequence.shape)
print("Encoded Graph Shape:", encoded_graph.shape)
print("Encoded Point Cloud Shape:", encoded_point_cloud.shape)
print(concatenated_data.shape)

with open('{}/fusion_STRING.pkl'.format(data_folder), 'wb') as f:
    pickle.dump(processed_data_list, f)
# with open('{}/fusion_Yeast.pkl'.format(data_folder), 'wb') as f:
#     pickle.dump(processed_data_list, f)
