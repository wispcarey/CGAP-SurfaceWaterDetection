import argparse
import numpy as np
import pandas as pd
import os
import time
import torch
import rasterio
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import graphlearning as gl

import ee

from ActiveLearning.adaptive_active_learning import adaptive_K
from datetime import datetime

from utils import set_loader_from_tif, set_model, laplace, class_accuracy, boundary_accuracy

from cli import parse_option_TestAndEvaluate

try:
    import yaml
    import openai
    import ee
    from GEE_utils import get_landsat_data, get_lon_lat

    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    OPEN_AI_API_KEY = config['OPEN_AI_API_KEY']
    GEE_PROJECT = config['GEE_PROJECT']
except:
    print("Warning: If you are running this for GEE data, please use Google Colab.")

# --test_dataset_path "RiverPIXELS/Patches/Colville_River_2 2015-07-11 076 011 L8 83 landsat.tif"
# --test_dataset_path "landsat_data/Landsat8_Image_LA.tif"
# --show_figures --test_dataset_path "landsat_data/Landsat5_Image_Rectangle_Region.tif" --random_sample

# Define the colors
cmap_colors = [(0, "green"), (0.5, "lightblue"), (1, "sienna")]

# Create a color map
RiverPIXELS_cmap = mcolors.LinearSegmentedColormap.from_list("RiverPIXELS", cmap_colors)

if __name__ == '__main__':
    modify_label = False
    opt = parse_option_TestAndEvaluate()
    suffix = ''

    np.random.seed(42)

    if opt.ori_cmap:
        cmap = 'viridis'
    else:
        cmap = RiverPIXELS_cmap

    npy_image_size = int(np.sqrt(opt.sample_batch_size))

    train_dataset = np.load(opt.train_dataset_path, allow_pickle=True).item()
    train_features = train_dataset['feature']
    train_labels = train_dataset['label']

    if modify_label:
        train_labels[train_labels == 2] = 0

    print("Number of training features:", len(train_labels))
    v,c = np.unique(train_labels, return_counts=True)
    print("Counts for 3 classes:", c)

    now = datetime.now()
    date_time_string = now.strftime("%Y-%m-%d %H:%M:%S")

    if opt.test_dataset_path == 'No':
        print("Using Google Earth Engine to Extract the dataset.")
        os.environ['OPENAI_API_KEY'] = OPEN_AI_API_KEY

        # ee initialization
        # ee.Authenticate()

        # Initialize the library.
        ee.Initialize(project=GEE_PROJECT)

        if opt.lon_lat == 'auto':
            print(f"Using ChatGPT to get the lon and lat value of the location {opt.location}")
            lon, lat, _ = get_lon_lat(opt.location)
        else:
            parts = opt.lon_lat.split(',')
            lon, lat = float(parts[0]), float(parts[1])
        print(f"Select the rectangle region centering at {lon}, {lat}.")

        gee_save_name = f"{opt.gee_save_name}_{date_time_string}"

        get_landsat_data(lon, lat,
                         dataset=opt.ee_dataset,
                         start_date=opt.startdate,
                         end_date=opt.enddate,
                         lonRange=opt.lon_range,
                         latRange=opt.lat_range,
                         bands=None,
                         display_image=opt.display_image,
                         save_data=True,
                         save_name=gee_save_name,
                         scale=opt.scale,
                         region=None,
                         folder='GEE_Landsat_Data_Download_Files',
                         )
        opt.test_dataset_path = f"GEE_Landsat_Data_Download_Files/{gee_save_name}.tif"
        print(
            f"Google Earth Engine data extraction finished. Wait {opt.checkfile_waiting_time} seconds for downloading.")
        time.sleep(opt.checkfile_waiting_time)

        for attempt in range(1, opt.checkfile_retries + 1):
            print(f"Checking for file, attempt {attempt}...")

            if os.path.exists(opt.test_dataset_path):
                print("File found.")
                break
            else:
                print("File not found.")

            if attempt < opt.checkfile_retries:
                print(f"Wait {opt.checkfile_waiting_time} seconds.")
                time.sleep(opt.checkfile_waiting_time)

        if not os.path.exists(opt.test_dataset_path):
            raise ValueError(f"File {opt.test_dataset_path} still not found after {opt.checkfile_retries} attempts.")

    file_path, file_extension = os.path.splitext(opt.test_dataset_path)

    if opt.figure_save_path == 'auto':
        opt.figure_save_path = f"{file_path}_processed_result.png"

    if file_extension == ".npy":
        test_dataset = np.load(opt.test_dataset_path, allow_pickle=True).item()
        all_test_features = test_dataset['feature']
        all_test_labels = test_dataset['label']

        # opt.sample_batch_size = 65536
        # opt.random_sample = False
    else:
        test_loader, tif_image = set_loader_from_tif(opt)
        model = set_model(opt)

        device = opt.dev

        all_test_features = None
        with torch.no_grad():
            for data, labels in test_loader:
                data = data.float()
                if all_test_features is None:
                    all_test_features = model(data.to(device)).cpu().numpy()
                    all_test_labels = labels.numpy()
                else:
                    all_test_features = np.concatenate((all_test_features, model(data.to(device)).cpu().numpy()))
                    all_test_labels = np.concatenate((all_test_labels, labels.numpy()))
    print("The input dataset is preprocessed with the embedding neural network.")

    if modify_label:
        all_test_labels[all_test_labels == 2] = 0

    N = len(all_test_features)
    if opt.random_sample:
        all_inds = np.random.permutation(N)
    else:
        all_inds = np.arange(N)

    num_batches = (N - 1) // opt.sample_batch_size + 1
    pred_labels = np.zeros(len(all_test_features))

    print(
        f"Number of test features: {N}. Divided into {num_batches} sample batches to do graph leanring with the maximum batch size {opt.sample_batch_size}.")
    if file_extension == ".npy" and (not opt.random_sample):
        bd3_counts_all, bd3_correct_all = 0, 0
        bd10_counts_all, bd10_correct_all = 0, 0
        class_counts_all, class_correct_all = np.zeros(3), np.zeros(3)
        class_Fcounts_all, class_FP_all = np.zeros(3), np.zeros(3)

    for s_batch_ind in range(num_batches):
        print(f"Start sample batch {s_batch_ind}.")
        t = time.time()
        start_ind = s_batch_ind * opt.sample_batch_size
        end_ind = min((s_batch_ind + 1) * opt.sample_batch_size, N)
        select_inds = all_inds[start_ind:end_ind]
        select_fvecs = all_test_features[select_inds]
        combine_fvecs = np.concatenate((train_features, select_fvecs), axis=0)
        labeled_inds = np.arange(len(train_labels))

        if opt.ssl_method == "Stable_Laplace":
            classification = laplace(combine_fvecs, train_labels, knn_num=50, epsilon='auto',
                                     n_classes='auto', tau=1e-8, similarity='angular')
            s_pred_labels = np.argmax(classification, axis=1)
        else:
            # W, k, knn_data = adaptive_K(combine_fvecs, k=25, verbose=False)

            W = gl.weightmatrix.knn(combine_fvecs, k=50, kernel='gaussian',
                                    eta=None, symmetrize=True, metric='raw', similarity='angular',
                                    knn_data=None)
            train_ind = np.arange(len(train_labels))
            class_priors = None

            if opt.ssl_method == "Laplace":
                model = gl.ssl.laplace(W, class_priors=class_priors)
            elif opt.ssl_method == "rw_Laplace":
                model = gl.ssl.laplace(W, class_priors, reweighting="poisson")
            elif opt.ssl_method == "Poisson":
                model = gl.ssl.poisson(W, class_priors)
            else:
                raise ValueError(f"Invalid choice of method {opt.ssl_method}.")
            classification = model.fit(train_ind, train_labels)
            s_pred_labels = model.predict()

            s_pred_labels = s_pred_labels[len(train_labels):]

        if np.isnan(classification).any():
            print("NAN value appear.")

        pred_labels[select_inds] = s_pred_labels
        if file_extension == ".npy" and (not opt.random_sample):
            acc = np.sum(s_pred_labels == all_test_labels[select_inds]) / len(select_inds)
            bd3_counts, bd3_correct = boundary_accuracy(s_pred_labels.reshape(npy_image_size, npy_image_size),
                                                        all_test_labels[select_inds].reshape(npy_image_size, npy_image_size),
                                                        d=3)
            bd10_counts, bd10_correct = boundary_accuracy(s_pred_labels.reshape(npy_image_size, npy_image_size),
                                                          all_test_labels[select_inds].reshape(npy_image_size, npy_image_size),
                                                          d=10)
            class_counts, class_correct, FP_counts = class_accuracy(s_pred_labels, all_test_labels[select_inds], return_FP=True)
            FP_all = np.sum(class_counts) - class_counts
            result_string = f"Time: {time.time() - t}; OA: {acc * 100: .2f}. " \
                            + f"\nBA(3): {bd3_correct / bd3_counts * 100: .2f}" \
                            + f"\nBA(10): {bd10_correct / bd10_counts * 100: .2f}" \
                            + f"\nTPR:" + ",".join(['{}: {:.2f}'.format(i, x / y * 100) for i, (x, y) in enumerate(zip(class_correct, class_counts))]) \
                            + f"\nFPR:" + ",".join(['{}: {:.2f}'.format(i, x / y * 100) for i, (x, y) in enumerate(zip(FP_counts, FP_all))])
            print(result_string)
            bd3_counts_all += bd3_counts
            bd3_correct_all += bd3_correct
            bd10_counts_all += bd10_counts
            bd10_correct_all += bd10_correct
            class_counts_all += class_counts
            class_correct_all += class_correct
            class_Fcounts_all += FP_all
            class_FP_all += FP_counts
            # print(bd3_counts_all, bd3_correct_all, bd10_counts_all, bd10_correct_all, class_counts_all, class_correct_all)
        else:
            print(f"Time: {time.time() - t}")
        if opt.show_figures and (not opt.random_sample) and file_extension == ".npy":
            img1 = s_pred_labels.reshape(npy_image_size, npy_image_size)
            img2 = all_test_labels[select_inds].reshape(npy_image_size, npy_image_size)

            if modify_label:
                img1[0, 0] = 2
                img2[0, 0] = 2

            fig_file_name, fig_extension = os.path.splitext(opt.test_dataset_path)

            img1_path = f"{fig_file_name}_pred_labels_{s_batch_ind}{suffix}.png"
            plt.imsave(img1_path, img1, cmap=cmap, format='png')

            img2_path = f"{fig_file_name}_gt_labels_{s_batch_ind}{suffix}.png"
            plt.imsave(img2_path, img2, cmap=cmap, format='png')

    if file_extension == ".npy":
        OA = np.sum(pred_labels == all_test_labels) / len(pred_labels)
        BA3 = bd3_correct_all / bd3_counts_all
        BA10 = bd10_correct_all / bd10_counts_all
        TPR = class_correct_all / class_counts_all
        FPR = class_FP_all / class_Fcounts_all
        result_vals = [OA, BA3, BA10, TPR[0], TPR[1], TPR[2], FPR[0], FPR[1], FPR[2]]
        result_vals = ['{:.2f}'.format(val * 100) for val in result_vals]
        data = {
            "Metric": ["OA", "BA(3)", "BA(10)", "TPR(0)", "TPR(1)",
                       "TPR(2)", "FPR(0)", "FPR(1)", "FPR(2)"],
            "Value": result_vals
        }
        df = pd.DataFrame(data)
        print(df)
    else:
        if opt.show_figures:
            rgb_img = tif_image[:, :, [2, 1, 0]] * 3
            reshape_label = pred_labels.reshape(rgb_img.shape[0], rgb_img.shape[1])

            fig, axs = plt.subplots(1, 2, figsize=(10, 5))
            axs[0].imshow(rgb_img, interpolation='none')
            axs[0].axis('off')
            axs[1].imshow(reshape_label, cmap=cmap, interpolation='none')
            axs[1].axis('off')
            plt.subplots_adjust(wspace=0.05, hspace=0)

            plt.savefig(opt.figure_save_path, bbox_inches='tight', dpi=300)
            plt.show()
