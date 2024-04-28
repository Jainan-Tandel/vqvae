import torch
from utils import CustomDataset

def save_latent_representations(model, checkpoint_path, dataset):
    vqvae_model = torch.load(checkpoint_path)
    model.load_state_dict(vqvae_model['model'])
    model.eval()  # Set model to evaluation mode

    encoder = model.encoder
    latent_path = dataset.save_discrete_latent_representations(encoder)
    return latent_path

