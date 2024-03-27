import matplotlib.pyplot as plt
import umap
from sklearn.manifold import TSNE
import torch
import numpy as np
import os

from torchvision import transforms, datasets
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score

from utils import set_loader, set_model
from cli import parse_option

import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from matplotlib.cm import viridis
from matplotlib.colors import Normalize

# --num_workers 4 --batch_size 2000 --margin 4 --model cnn --method Val --stepsize 4 --dataset_path RiverPIXELS/train_RiverPIXELS.npy --cp_load_path save/SupCon/SupCon_temp/ckpt_epoch_75.pth --seed 3 --down_sample_rate None

# Define the colors
cmap_colors = [(0, "green"), (0.5, "lightblue"), (1, "sienna")]

# Create a color map
RiverPIXELS_cmap = mcolors.LinearSegmentedColormap.from_list("RiverPIXELS", cmap_colors)


def umap_visualization(encoded_data, labels, data_name='', n_neighbors=20, sel_labels=None, base=None, base_labels=None,
                       save_dir=None):
    reducer = umap.UMAP(n_neighbors=n_neighbors)
    umap_embedded_data = reducer.fit_transform(encoded_data)

    plt.figure(figsize=(10, 10))

    scatter = plt.scatter(umap_embedded_data[:, 0], umap_embedded_data[:, 1], c=labels, cmap='viridis', s=.5)
    norm = Normalize(vmin=min(labels), vmax=max(labels))

    # class_names = ["Land", "Water", "Sediment"]
    # legend_elements = [
    #     Line2D([0], [0], marker='o', color='w', markerfacecolor=viridis(norm(i)), markersize=10, label=class_names[i]) for i
    #     in range(min(labels), max(labels) + 1)]
    # plt.legend(handles=legend_elements, loc="best")
    if sel_labels is not None:
        plt.scatter(umap_embedded_data[sel_labels, 0], umap_embedded_data[sel_labels, 1], c='r', edgecolors='black',
                    s=1, marker='*', cmap=RiverPIXELS_cmap)
    if base is not None:
        b = reducer.transform(base)
        plt.scatter(b[:, 0], b[:, 1], c=base_labels, edgecolors='red', s=20, marker='*', linewidth=0.5)
    # plt.title("UMAP Embedding of " + data_name + " Data")
    plt.axis('off')
    if save_dir is not None:
        if opt.cp_load_path == 'no':
            save_path = "UMAP_no.png"
        else:
            save_path = "{}_UMAP_{}.png".format(save_dir, data_name)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print("UMap visualization finished.")
    plt.close()


def tsne_visualization(encoded_data, labels, data_name='', sel_labels=None, base=None, base_labels=None, save_dir=None):
    if base is not None:
        encoded_data = np.concatenate((encoded_data, base), axis=0)
        l = base.shape[0]
    tsne_embedded_data = TSNE(n_components=2, init='random').fit_transform(encoded_data)
    plt.figure(figsize=(10, 10))
    if base is None:
        plt.scatter(tsne_embedded_data[:, 0], tsne_embedded_data[:, 1], c=labels, cmap='viridis', s=.5)
    norm = Normalize(vmin=min(labels), vmax=max(labels))

    # class_names = ["Land", "Water", "Sediment"]
    # legend_elements = [
    #     Line2D([0], [0], marker='o', color='w', markerfacecolor=viridis(norm(i)), markersize=10, label=class_names[i])
    #     for i
    #     in range(min(labels), max(labels) + 1)]
    # plt.legend(handles=legend_elements, loc="best")
    if sel_labels is not None:
        plt.scatter(tsne_embedded_data[sel_labels, 0], tsne_embedded_data[sel_labels, 1], c='r', edgecolors='black',
                    s=1, marker='*', cmap=RiverPIXELS_cmap)
    if base is not None:
        plt.scatter(tsne_embedded_data[:-l, 0], tsne_embedded_data[:-l, 1], c=labels, s=.5)
        plt.scatter(tsne_embedded_data[-l:, 0], tsne_embedded_data[-l:, 1], c=base_labels, edgecolors='red', s=20,
                    marker='*', linewidth=0.5)
    # plt.title("TSNE Embedding of " + data_name + " Data")
    plt.axis('off')
    if save_dir is not None:
        if opt.cp_load_path == 'no':
            save_path = "TSNE_no.png"
        else:
            save_path = "{}_TSNE_{}.png".format(save_dir, data_name)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print("TSNE visualization finished.")
    plt.close()

def visualize(opt, TSNE=True, save_dir=None, data_name="RiverPIXELS"):
    TSNE = opt.TSNE
    if TSNE:
        print("Use both UMAP and TSNE embedding.")
    else:
        print("Use only UMAP embedding.")

    model = set_model(opt)

    val_loader = set_loader(opt)
    device = opt.dev

    encoded_data = None
    with torch.no_grad():
        for data, labels in val_loader:
            data = data.float()
            if encoded_data is None:
                encoded_data = model(data.to(device)).cpu().numpy()
                all_labels = labels.numpy()
            else:
                encoded_data = np.concatenate((encoded_data, model(data.to(device)).cpu().numpy()))
                all_labels = np.concatenate((all_labels, labels.numpy()))

    print(all_labels.shape)
    print(encoded_data.shape)

    umap_visualization(encoded_data, all_labels, data_name=data_name, base=None, base_labels=None,
                       save_dir=save_dir)
    if TSNE:
        tsne_visualization(encoded_data, all_labels, data_name=data_name, base=None, base_labels=None,
                           save_dir=save_dir)
    plt.show()

if __name__ == '__main__':
    opt = parse_option()
    opt.method = 'Val'
    folder_path = os.path.dirname(opt.cp_load_path) + "/"
    file_name = os.path.basename(opt.cp_load_path)
    file_name, _ = os.path.splitext(file_name)
    print("data name", f"{file_name}")
    print("save dir", folder_path)
    visualize(opt, save_dir=folder_path, data_name=f"{file_name}")