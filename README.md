# ShapeFromShading
A convolutional neural network that takes an image as imput and produces a depth map. The network is trained on the RGB-D object dataset from University of Washington ([http://rgbd-dataset.cs.washington.edu/dataset/](http://rgbd-dataset.cs.washington.edu/dataset/)).
![](shape_from_shading.png)

## Minimum Requirements
- Python 3.6
- Tensorflow 1.x
- Pillow 8.0.1
- Open3D 0.11.2

## Test An Image
    python3 test_image.py <path_to_input_image> <path_to_weights>
Example using the trained weights provided with the code:
    
    python3 test_image.py ~/apple.png weights.h5py

## Training The Network Yourself
1. Download the rgbd object dataset from [here](http://rgbd-dataset.cs.washington.edu/dataset.html). (Get the zip file of the cropped objects and extract it)
2. Creating the h5py dataset from the rgbd dataset: (Warning: the dataset file can take more than 50 GB of space)

    `python3 create_dataset.py <path_to_rgbd_dataset_folder> <path_to_h5py_output_file>`
    
    Example:
    
    `python3 create_dataset.py ~/Downloads/rgbd-dataset dataset.h5py`
    
3. Train the network on the dataset:

    `python3 train.py <path_to_dataset_file> <path_to_output_weights_file>`
    
    Example:
    
    `python3 train.py dataset.h5py weights.h5py`

