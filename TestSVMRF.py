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

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

# --test_dataset_path "RiverPIXELS/Patches/Colville_River_2 2015-07-11 076 011 L8 83 landsat.tif"
# --test_dataset_path "landsat_data/Landsat8_Image_LA.tif"
# --show_figures --test_dataset_path "landsat_data/Landsat5_Image_Rectangle_Region.tif" --random_sample

# Define the colors
cmap_colors = [(0, "green"), (0.5, "lightblue"), (1, "sienna")]

# Create a color map
RiverPIXELS_cmap = mcolors.LinearSegmentedColormap.from_list("RiverPIXELS", cmap_colors)

def sample_indices_by_value(arr, sample_num, random_seed=42):
    np.random.seed(random_seed)
    unique_values = np.unique(arr)
    sampled_indices = []

    for value in unique_values:
        indices = np.where(arr == value)[0]
        if len(indices) <= sample_num:
            sampled_indices.extend(indices)
        else:
            sampled_indices.extend(np.random.choice(indices, sample_num, replace=False))

    return sampled_indices

if __name__ == '__main__':
    clf_method = 'SVM'
    suffix = '2class'
    n_classes = 2

    opt = parse_option_TestAndEvaluate()

    np.random.seed(42)

    train_dataset = np.load(opt.train_dataset_path, allow_pickle=True).item()
    train_features_all = train_dataset['feature']
    train_labels_all = train_dataset['label']

    num_train_batches = (len(train_features_all) - 1) // opt.sample_batch_size + 1
    for s_batch_ind in range(num_train_batches):
        start_ind = s_batch_ind * opt.sample_batch_size
        end_ind = min((s_batch_ind + 1) * opt.sample_batch_size, len(train_features_all))

        temp_features = train_features_all[start_ind:end_ind]
        temp_labels = train_labels_all[start_ind:end_ind]

        sampled_indices = sample_indices_by_value(temp_labels, sample_num=500, random_seed=42)

        if s_batch_ind == 0:
            train_features = temp_features[sampled_indices]
            train_labels = temp_labels[sampled_indices]
        else:
            train_features = np.concatenate((train_features, temp_features[sampled_indices]), axis=0)
            train_labels = np.concatenate((train_labels, temp_labels[sampled_indices]), axis=0)

    ## label change
    if n_classes == 2:
        train_labels[train_labels == 2] = 0

    print("Number of training features:", len(train_labels))
    v,c = np.unique(train_labels, return_counts=True)
    print("Counts for classes:", c)

    t = time.time()
    if clf_method == 'SVM':
        ## train SVM
        model_clf = SVC(C=1.0, kernel='rbf', degree=3, gamma='scale')
        model_clf.fit(train_features, train_labels)
    else:
        ## train RF
        model_clf = RandomForestClassifier(n_estimators=100, max_depth=None, min_samples_split=2, min_samples_leaf=1,
                                        random_state=42)
        model_clf.fit(train_features, train_labels)
    print(f"Finish training the {clf_method} classifier with the time {time.time() - t}.")

    file_path, file_extension = os.path.splitext(opt.test_dataset_path)

    test_dataset = np.load(opt.test_dataset_path, allow_pickle=True).item()
    all_test_features = test_dataset['feature']
    all_test_labels = test_dataset['label']

    ## label change
    if n_classes == 2:
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

        s_pred_labels = model_clf.predict(select_fvecs)

        pred_labels[select_inds] = s_pred_labels
        if file_extension == ".npy" and (not opt.random_sample):
            acc = np.sum(s_pred_labels == all_test_labels[select_inds]) / len(select_inds)
            bd3_counts, bd3_correct = boundary_accuracy(s_pred_labels.reshape(256, 256),
                                                        all_test_labels[select_inds].reshape(256, 256),
                                                        d=3)
            bd10_counts, bd10_correct = boundary_accuracy(s_pred_labels.reshape(256, 256),
                                                          all_test_labels[select_inds].reshape(256, 256),
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
            img1 = s_pred_labels.reshape(256, 256)
            img2 = all_test_labels[select_inds].reshape(256, 256)

            if n_classes == 2:
                img1[0,0] = 2
                img2[0,0] = 2

            # plt.imshow(img1)
            # plt.axis('off')

            fig_file_name, fig_extension = os.path.splitext(opt.test_dataset_path)
            batch_figure_path = f"{fig_file_name}_{s_batch_ind}_{clf_method}{suffix}.png"
            plt.imsave(batch_figure_path, img1, format='png')

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

