import numpy as np
from utils import *

if __name__ == "__main__":
    data_path = "RiverPIXELS/test_RiverPIXELS.npy"
    test_data = np.load(data_path, allow_pickle=True).item()
    test_data_128 = {}
    test_data_512 = {}

    for i in range(10):
        image, label = test_data['test']['image'][i], test_data['test']['label'][i]
        image_128, label_128 = resize_image_and_labels(image, label, output_size=(128, 128))
        image_512, label_512 = resize_image_and_labels(image, label, output_size=(512, 512))
        if i == 0:
            image_all_128, label_all_128 = image_128[np.newaxis,:,:,:], label_128[np.newaxis,:,:]
            image_all_512, label_all_512 = image_512[np.newaxis,:,:,:], label_512[np.newaxis,:,:]
        else:
            image_all_128, label_all_128 = np.concatenate((image_all_128, image_128[np.newaxis,:,:,:]), axis=0), \
                                           np.concatenate((label_all_128, label_128[np.newaxis,:,:]), axis=0)
            image_all_512, label_all_512 = np.concatenate((image_all_512, image_512[np.newaxis,:,:,:]), axis=0), \
                                           np.concatenate((label_all_512, label_512[np.newaxis,:,:]), axis=0)

    test_data_128['test'] = {'image': image_all_128, 'label': label_all_128}
    test_data_512['test'] = {'image': image_all_512, 'label': label_all_512}

    np.save("RiverPIXELS/test_RiverPIXELS_128.npy", test_data_128)
    np.save("RiverPIXELS/test_RiverPIXELS_512.npy", test_data_512)

    # data_path = "RiverPIXELS/test_RiverPIXELS_128.npy"
    # test_data = np.load(data_path, allow_pickle=True).item()
    # print(test_data['test']['image'].shape)

