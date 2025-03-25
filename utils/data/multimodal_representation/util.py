import warnings
import random
import os
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import LabelEncoder
from torch_geometric.nn import TopKPooling
from torch_geometric.utils import negative_sampling
from torch_geometric.data import Data
from Bio.PDB import PDBParser, PPBuilder, Polypeptide


from model.protein_feature_generator.pointcloud import *

warnings.filterwarnings("ignore")
random.seed(0)
parser = PDBParser(QUIET=True)
ppb = PPBuilder()

current_file_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Check if GPU is available
if torch.cuda.is_available():
    device = torch.device('cuda')
    print("Using GPU:", torch.cuda.get_device_name(0))
else:
    device = torch.device('cpu')
    print("No GPU available, using CPU.")

amino_acids = 'ACDEFGHIKLMNPQRSTVWYX'
label_encoder = LabelEncoder()
label_encoder.fit(list(amino_acids))
num_amino_acids = len(amino_acids)


def one_hot_encode_amino_acid(sequence=None, amino_acid_indices=None):
    amino_acid_indices = label_encoder.transform(list(sequence))
    one_hot = np.zeros((len(sequence), num_amino_acids), dtype=np.float32)
    one_hot[np.arange(len(sequence)), amino_acid_indices] = 1
    return one_hot


def z_score_standardization(tensor):
    mean = tensor.mean()
    std = tensor.std()
    if std != 0:
        standardized_tensor = (tensor - mean) / std
    else:
        standardized_tensor = tensor  # Handle the case when std is 0
    return standardized_tensor


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

    return processed_encoded_graph[:fixed_size]


def read_pdb(pdb_path):
    structure = parser.get_structure('protein', pdb_path)

    # Graph
    coordinates = []
    sequence = ""
    k_neighbors = 5
    for residue in structure.get_residues():
        if 'CA' in residue:
            try:
                aa_code = Polypeptide.three_to_one(residue.get_resname())
            except KeyError:
                aa_code = "X"
            sequence += aa_code
            coordinates.append(residue['CA'].get_coord())
    coordinates = np.array(coordinates, dtype=np.float32)
    node_features = one_hot_encode_amino_acid(sequence)
    x = torch.tensor(node_features, dtype=torch.float32)
    edge_index = kneighbors_graph(coordinates, k_neighbors, mode='connectivity', include_self=False)
    edge_index = edge_index.nonzero()
    edge_index = np.array(edge_index)
    edge_index = torch.from_numpy(edge_index).to(torch.long).contiguous()
    neg_edge_index = negative_sampling(
        edge_index=edge_index,
        num_nodes=x.size(0),
        num_neg_samples=edge_index.size(1) // 2
    )
    graph = Data(x=x, edge_index=edge_index, neg_edge_index=neg_edge_index)

    # Point Cloud
    coordinates = []
    desired_num_points = 2048
    for atom in structure.get_atoms():
        coordinates.append(atom.get_coord())
    coordinates = np.array(coordinates, dtype=np.float32)
    num_points = coordinates.shape[0]
    if num_points < desired_num_points:
        padding = np.zeros((desired_num_points - num_points, 3), dtype=np.float32)
        coordinates = np.concatenate((coordinates, padding), axis=0)
    elif num_points > desired_num_points:
        coordinates = coordinates[:desired_num_points, :]
    coordinates = torch.tensor(coordinates, dtype=torch.float32)
    coordinates -= coordinates.mean(0)
    d = np.sqrt((coordinates ** 2).sum(1))
    coordinates /= d.max()
    point_cloud = torch.FloatTensor(coordinates).permute(1, 0)

    return graph, point_cloud


def get_multimodal_representation(protein_path, vae_model, VGAE, PAE, Fusion, sequence):
    graph, point_cloud = read_pdb(protein_path)

    # Using GPU
    sequence = sequence.to(device)
    graph = graph.to(device)
    point_cloud = point_cloud.to(device)

    with torch.no_grad():
        sequence_STRING = sequence.unsqueeze(0)
        encoded_sequence = vae_model.encoder.get_encoder_output(sequence_STRING).to("cpu")
        encoded_sequence = z_score_standardization(encoded_sequence)

    # Pass the graph data through VGAE for encoding
    with torch.no_grad():
        encoded_graph = VGAE.encode(graph.x, graph.edge_index).to("cpu")
        encoded_graph = process_encoded_graph(encoded_graph, graph.edge_index.to("cpu"))
        encoded_graph = torch.mean(encoded_graph, dim=1)
        encoded_graph = z_score_standardization(encoded_graph)

    # Pass the point cloud data through PAE for encoding
    with torch.no_grad():
        encoded_point_cloud = PAE.encode(point_cloud[None, :]).squeeze().to("cpu")
        encoded_point_cloud = z_score_standardization(encoded_point_cloud)

    concatenated_data = torch.cat((encoded_sequence, encoded_graph, encoded_point_cloud), dim=0).unsqueeze(0).to(device)
    multimodal_representation = Fusion.encode(concatenated_data).squeeze().to("cpu")
    return multimodal_representation
