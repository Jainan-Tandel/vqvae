import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import time
import os
from datasets.block import BlockDataset, LatentBlockDataset
import numpy as np
from PIL import Image


class CustomDataset(Dataset):
    def __init__(self, img_dir="data", train=True, transform=None,size=(32,32)):
        self.size=(*size,3)
        folder = "/Train_data/" if train==True else "/Test_data/"
        self.img_dir = img_dir+folder
        self.img_filenames = [x for x in os.listdir(self.img_dir) if x.endswith(".jpg") or x.endswith(".JPG")]
        # self.label_df = pd.read_csv(label_file)  # Assuming labels are stored in a CSV file
        self.transform = transform

    def __len__(self):
        return len(self.img_filenames)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.img_filenames[idx]  # Assuming first column contains image filenames
        img_path = f'{self.img_dir}/{img_name}'
        image = Image.open(img_path)

        if self.transform:
            image = self.transform(image)

        # Get one-hot encoded labels from the dataframe
        return image, torch.tensor([1])
    def find_var(self):
        all_images = np.zeros(shape=[len(self),*self.size])
        for idx in range(len(self)):
            img_name = self.img_filenames[idx]  # Assuming first column contains image filenames
            img_path = f'{self.img_dir}/{img_name}'
            image = Image.open(img_path)
            if self.transform:
                image = transforms.Resize(size=self.size[:2])(transforms.ToTensor()(image))
            all_images[idx] = np.array(image.permute((1,2,0)))
        return np.var(all_images)

def load_isic():
    data_folder_path = os.getcwd()
    data_file_path = data_folder_path +\
        '/data'
    train = CustomDataset(img_dir=data_file_path, train=True, transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Resize(size=(32,32)),
                               transforms.Normalize(
                                   (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                           ]))
    val = CustomDataset(img_dir=data_file_path, train=False, transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Resize(size=(32,32)),
                               transforms.Normalize(
                                   (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                           ]))
    return train, val

def load_cifar():
    train = datasets.CIFAR10(root="data", train=True, download=True,
                             transform=transforms.Compose([
                                 transforms.ToTensor(),
                                 transforms.Normalize(
                                     (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                             ]))

    val = datasets.CIFAR10(root="data", train=False, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize(
                                   (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                           ]))
    return train, val


def load_block():
    data_folder_path = os.getcwd()
    data_file_path = data_folder_path + \
        '/data/randact_traj_length_100_n_trials_1000_n_contexts_1.npy'

    train = BlockDataset(data_file_path, train=True,
                         transform=transforms.Compose([
                             transforms.ToTensor(),
                             transforms.Normalize(
                                 (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                         ]))

    val = BlockDataset(data_file_path, train=False,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize(
                               (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                       ]))
    return train, val

def load_latent_block():
    data_folder_path = os.getcwd()
    data_file_path = data_folder_path + \
        '/data/latent_e_indices.npy'

    train = LatentBlockDataset(data_file_path, train=True,
                         transform=None)

    val = LatentBlockDataset(data_file_path, train=False,
                       transform=None)
    return train, val


def data_loaders(train_data, val_data, batch_size):

    train_loader = DataLoader(train_data,
                              batch_size=batch_size,
                              shuffle=True,
                              pin_memory=True)
    val_loader = DataLoader(val_data,
                            batch_size=batch_size,
                            shuffle=True,
                            pin_memory=True)
    return train_loader, val_loader


def load_data_and_data_loaders(dataset, batch_size):
    if dataset =='ISIC':
        training_data, validation_data = load_isic()
        training_loader, validation_loader = data_loaders(
            training_data, validation_data, batch_size)
        # x_train_var = np.var(training_data / 1.0)
        x_train_var = training_data.find_var()
    elif dataset == 'CIFAR10':
        training_data, validation_data = load_cifar()
        training_loader, validation_loader = data_loaders(
            training_data, validation_data, batch_size)
        x_train_var = np.var(training_data.train_data / 1.0)

    elif dataset == 'BLOCK':
        training_data, validation_data = load_block()
        training_loader, validation_loader = data_loaders(
            training_data, validation_data, batch_size)

        x_train_var = np.var(training_data.data / 1.0)
    elif dataset == 'LATENT_BLOCK':
        training_data, validation_data = load_latent_block()
        training_loader, validation_loader = data_loaders(
            training_data, validation_data, batch_size)

        x_train_var = np.var(training_data.data)

    else:
        raise ValueError(
            'Invalid dataset: only CIFAR10 and BLOCK datasets are supported.')

    return training_data, validation_data, training_loader, validation_loader, x_train_var


def readable_timestamp():
    return time.ctime().replace('  ', ' ').replace(
        ' ', '_').replace(':', '_').lower()


def save_model_and_results(model, results, hyperparameters, timestamp):
    SAVE_MODEL_PATH = os.getcwd() + '/results'

    results_to_save = {
        'model': model.state_dict(),
        'results': results,
        'hyperparameters': hyperparameters
    }
    torch.save(results_to_save,
               SAVE_MODEL_PATH + '/vqvae_data_' + timestamp + '.pth')
