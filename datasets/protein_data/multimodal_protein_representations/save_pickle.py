import pandas as pd
import pickle
import os

current_file_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

csv_name = '{}/raw_data/STRING_AF2DB.csv'.format(current_file_path)
# csv_name = '{}/raw_data/Yeast_AF2DB.csv'.format(current_file_path)

df = pd.read_csv(csv_name, header=None)

pkl_path = '{}/protein_data/multimodal_protein_representations/multimodal_STRING.pkl'.format(current_file_path)
# pkl_path = '{}/protein_data/multimodal_protein_representations/multimodal_Yeast.pkl'.format(current_file_path)

with open(pkl_path, 'rb') as file:
    data = pickle.load(file)

save_path = '{}/protein_data/multimodal_protein_representations/all_protein_STRING.pickle'.format(current_file_path)
# save_path = '{}/protein_data/multimodal_protein_representations/all_protein_Yeast.pickle'.format(current_file_path)

result_dict = {'.'.join(row[0].split('.')[:-1]): data[i] for i, row in enumerate(df.itertuples(index=False))}

with open(save_path, 'wb') as file:
    pickle.dump(result_dict, file)

print("Dictionary saved successfully.")
