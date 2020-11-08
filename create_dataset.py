import numpy as np
import os
from PIL import Image
import h5py
import random
import sys

dataset_path = sys.argv[1]
category_dirs = os.scandir(dataset_path)

file_paths_dict = {}
file_suffixes = ['_crop.png', '_depthcrop.png', '_maskcrop.png']
for category_dir in category_dirs:
    category_path = dataset_path + category_dir.name + '/'
    sequence_dirs = os.scandir(category_path)
    for sequence_dir in sequence_dirs:
        sequence_path = category_path + sequence_dir.name + '/'
        sequence_files = os.scandir(sequence_path)
        for sequence_file in sequence_files:
            file_name = sequence_file.name
            file_path = sequence_path + file_name
            # print(file_name)
            for i in range(3):
                if file_name.endswith(file_suffixes[i]):
                    prefix = file_name[:len(file_name) - len(file_suffixes[i])]
                    if prefix not in file_paths_dict:
                        file_paths_dict[prefix] = ['', '', '']
                    file_paths_dict[prefix][i] = file_path

file_paths_list = []
num_skipped = 0
# remove samples that are not complete
for name in file_paths_dict:
    print(name)
    file_paths = file_paths_dict[name]
    keep_it = True
    for i in range(3):
        if not file_paths[i]:
            keep_it = False
    if keep_it:
        file_paths_list.append(file_paths)
    else:
        num_skipped += 1

print("num_skipped")
print(num_skipped)

# shuffle
random.seed(1)
random.shuffle(file_paths_list)

# prepare dataset file
d_file = h5py.File(sys.argv[2], 'w')
input_set = d_file.create_dataset("input", (207662, 128, 128, 3), 'f')
target_set = d_file.create_dataset("target", (207662, 128, 128, 1), 'f')

# read and add the images to the dataset
set_size = 207645
set_index = 0
for file_paths in file_paths_list:
    print(set_index)

    curr_images = []
    for i in range(3):
        image = Image.open(file_paths[i]).resize((128, 128), Image.NEAREST)
        curr_images.append(np.array(image))

    image = np.array(curr_images[0])/255
    depth = np.array(curr_images[1])
    # mask = np.array(curr_images[2])
    # mask = np.logical_and(mask, depth != 0)

    mask = (depth != 0)

    min_depth = np.amin(depth[mask])
    max_depth = np.amax(depth[mask])

    if (max_depth-min_depth) < 10:
        continue

    # print("min: "+str(min_depth))
    # print("max; "+str(max_depth))
    #
    # valid_depths = depth[mask]
    # mean_depth = np.mean(valid_depths)
    # std_depth = np.std(valid_depths)
    #
    # print("mean: "+str(mean_depth))
    # print("std: "+str(std_depth))
    # exit()

    depth = (depth-min_depth)/(max_depth-min_depth)

    depth[~mask] = -1

    input_set[set_index, ...] = image
    target_set[set_index, :, :, 0] = depth
    set_index += 1

print("total tupples")
print(len(file_paths_list))

print('total in dataset')
print(set_index)

exit()
