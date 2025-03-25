import pandas as pd
import pickle
from tqdm import tqdm

from utils.data.multimodal_representation.util import *
from utils.data.sequence import get_feature_origin

# Load pre-trained models
vae_model = torch.load("{}/datasets/protein_data/model_checkpoints/Seq_VAE_STRING.pt".format(current_file_path), map_location=device)
vgae_model = torch.load("{}/datasets/protein_data/model_checkpoints/VGAE_STRING.pt".format(current_file_path), map_location=device)
pae_model = torch.load("{}/datasets/protein_data/model_checkpoints/PAE_STRING.pt".format(current_file_path), map_location=device)
fusion_model = torch.load("{}/datasets/protein_data/model_checkpoints/Fusion_STRING.pt".format(current_file_path), map_location=device)
# vae_model = torch.load("{}/datasets/protein_data/model_checkpoints/Seq_VAE_Yeast.pt".format(current_file_path), map_location=device)
# vgae_model = torch.load("{}/datasets/protein_data/model_checkpoints/VGAE_Yeast.pt".format(current_file_path), map_location=device)
# pae_model = torch.load("{}/datasets/protein_data/model_checkpoints/PAE_Yeast.pt".format(current_file_path), map_location=device)
# fusion_model = torch.load("{}/datasets/protein_data/model_checkpoints/Fusion_Yeast.pt".format(current_file_path), map_location=device)
print("Pre-trained models loaded successfully.")

data_folder = "{}/datasets/raw_data".format(current_file_path)
# Read the label CSV file
df = pd.read_csv('{}/STRING_AF2DB.csv'.format(data_folder), header=None)
# df = pd.read_csv('{}/Yeast_AF2DB.csv'.format(data_folder), header=None)
print("Number of samples:", len(df))

mulmodal = []
sequence = []
graph = []
point_cloud = []

protein_vec_list = get_feature_origin('{}/datasets/raw_data/STRING_sequence.tsv'.format(current_file_path), '{}/datasets/raw_data/vec5_CTC.txt'.format(current_file_path))
# protein_vec_list = get_feature_origin('{}/datasets/raw_data/Yeast_sequence.tsv'.format(current_file_path), '{}/datasets/raw_data/vec5_CTC.txt'.format(current_file_path))

sequence_STRING = []
for vec in protein_vec_list:
    sequence_STRING.append(vec)
sequence_STRING = np.array(sequence_STRING)
sequence_STRING = torch.tensor(sequence_STRING, dtype=torch.float)

# Iterate through the dataset to process each sample
for index, protein_name in tqdm(df.iterrows(), total=df.shape[0]):
    pdb_path = "{}/STRING_AF2DB/{}".format(data_folder, protein_name[0])
    # pdb_path = "{}/Yeast_AF2DB/{}".format(data_folder, protein_name[0])
    multimodal_representation = get_multimodal_representation(pdb_path, vae_model, vgae_model, pae_model, fusion_model, sequence_STRING[index])

    # Concatenate multimodal representation with ligand representation
    mulmodal_feature = multimodal_representation.detach().numpy()

    # Append the features to their respective lists
    mulmodal.append(mulmodal_feature)

# Save the features to pickle files
with open('{}/datasets/protein_data/multimodal_protein_representations/multimodal_STRING.pkl'.format(current_file_path), 'wb') as f:
    pickle.dump(mulmodal, f)
# with open('{}/datasets/protein_data/multimodal_protein_representations/multimodal_Yeast.pkl'.format(current_file_path), 'wb') as f:
#     pickle.dump(mulmodal, f)

print("saved")
