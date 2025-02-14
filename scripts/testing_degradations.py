import torch
import timm
import torchvision.transforms as T
from wildlife_datasets.datasets import WildlifeReID10k, WildlifeDataset
from wildlife_tools.data import WildlifeDataset
from wildlife_tools.features import DeepFeatures
from lib.experimentClass import ExpSetup
import os
from wildlife_tools.inference import KnnClassifier
from wildlife_tools.similarity import CosineSimilarity
import numpy as np
import pandas as pd
from degradations import RandomMotionBlur, GGPSFBlur, JPEGArtifacts, GaussianNoise


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
img_size = (384, 384)
MODEL_PATH = "xdg_new/checkpoint99.pth"
batch_size=4
# num_workers = int(os.getenv("NSLOTS")) // torch.cuda.device_count() - 1
num_workers=4
exp = ExpSetup()

root = os.path.join(exp.work_dir, f'Datasets_{img_size[0]}')
dataset = WildlifeReID10k(root, load_label=True)
dataset.df = dataset.df.drop('date', axis=1)

idx_train = dataset.df['split'] == 'train'
idx_database = dataset.df['split'] == 'database'
idx_query = dataset.df['split'] == 'query'

idx_train = dataset.df[idx_train].index
idx_database = dataset.df[idx_database].index
idx_query = dataset.df[idx_query].index

combined_idx = idx_train.union(idx_database)

transform = T.Compose([
    T.Resize(size=img_size),
    T.ToTensor(),
    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])

dataset_database = WildlifeDataset(
    metadata = dataset.df.loc[combined_idx], 
    root = root,
    transform=transform
)

training_database = WildlifeDataset(
    metadata = dataset.df.loc[idx_train], 
    root = root,
    transform=transform
)

df_query=dataset.df.loc[idx_query]
df_database_only = dataset.df.loc[idx_database]

degradations = [
    ['gaussian blur', T.GaussianBlur, [1, 3, 7, 11, 17], 10],
    ['motion blur', RandomMotionBlur, [4, 8, 12, 16, 20], None],
    ['out-of-focus blur', GGPSFBlur, [5, 9, 13, 17], 0.5],
    ['jpeg artifacts', JPEGArtifacts, [90, 70, 40, 10, 5], None],
]

# Backbone and loss configuration
# model = timm.create_model('swin_large_patch4_window12_384', num_classes=0, pretrained=True)
# checkpoint =  torch.load(os.path.join(exp.model_dir, MODEL_PATH))
# model.load_state_dict(checkpoint['model'])
model = timm.create_model("hf-hub:BVRA/wildlife-mega-L-384", pretrained=True)
model = model.to(device)
model.eval()

results_df = pd.DataFrame(columns=['species', 'degradation', 'parameter', 'accuracy'])

for degradation in degradations:
    deg_name = degradation[0]
    deg_dunction = degradation[1]
    deg_params_it = degradation[2]
    deg_params_set = degradation[3]
    
    for param in deg_params_it:
        if deg_params_set:
            transform = T.Compose([
                T.Resize(size=img_size),
                T.ToTensor(),
                deg_dunction(param, deg_params_set),
                GaussianNoise(mean=0, std=0.05, p=1),
                T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ])
        else:
            transform = T.Compose([
                T.Resize(size=img_size),
                T.ToTensor(),
                deg_dunction(param),
                GaussianNoise(mean=0, std=0.05, p=1),
                T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ])

        query_dataset = WildlifeDataset(
            metadata = df_query, 
            root = root,
            transform=transform
        )

        df_query_filtered = df_query[~df_query['identity'].isin(df_database_only['identity'])]
        assert df_query_filtered['identity'].isin(df_database_only['identity']).sum() == 0, "Some identities from df1 are still in df2_filtered"


        filtered_dataset = WildlifeDataset(
            metadata = df_query_filtered, 
            root = root,
            transform=transform
        )

        species_list = query_dataset.metadata['dataset'].unique()  # Get unique species
        accuracy_per_species = {}
        for species in species_list:
            species_indices = query_dataset.metadata['dataset'] == species


        query_input = filtered_dataset
        dataset_input = training_database


        extractor = DeepFeatures(model, batch_size=batch_size, num_workers=num_workers, device=device)
        query, database = extractor(query_input), extractor(dataset_input)

        similarity_function = CosineSimilarity()
        similarity = similarity_function(query, database)

        classifier = KnnClassifier(k=1, database_labels=dataset_input.labels_string)
        predictions = classifier(similarity['cosine'])
        per_sample_correctness = query_input.labels_string == predictions
        accuracy = np.mean(per_sample_correctness)
        print(f"total accuracy: {accuracy*100:.4f}")

        predictions_df = query_input.metadata.copy()
        predictions_df['predicted_labels'] = predictions
        predictions_df.to_csv(f"{exp.out_dir}/predictions_filtered_pretrained_md.csv")

        species_list = query_input.metadata['dataset'].unique()
        accuracy_per_species = {}

        for species in species_list:
            species_indices = query_input.metadata['dataset'] == species

            # Filter labels and predictions for the current species
            species_labels = query_input.labels_string[species_indices]
            species_predictions = predictions[species_indices]

            species_accuracy = (species_labels == species_predictions).mean()*100
            accuracy_per_species[species] = species_accuracy

            if param == 1:
                results_df = results_df.append({'species': species, 'degradation': 'none', 'parameter': 'none', 'accuracy': species_accuracy}, ignore_index=True)
            else:
                results_df = results_df.append({'species': species, 'ks': param, 'accuracy': species_accuracy}, ignore_index=True)

        # Print the accuracies per species
        for species, acc in accuracy_per_species.items():
            print(f"Accuracy for species {species}: {acc:.4f}")

# Export the results df - not sure where
#results_df.to_csv()