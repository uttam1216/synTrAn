# DCGAN in Tensorflow

This folder includes the DCGAN model in __TrajGen__.

## Prerequisites

- Python 2.7 or Python 3.3+
- [Tensorflow 0.12.1](https://github.com/tensorflow/tensorflow/tree/r0.12)
- [SciPy](http://www.scipy.org/install.html)
- [pillow](https://github.com/python-pillow/Pillow)
- (Optional) [moviepy](https://github.com/Zulko/moviepy) (for visualization)
- (Optional) [Align&Cropped Images.zip](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) : Large-scale CelebFaces Dataset


## Usage

To train a model with downloaded dataset:

    $ python main.py --dataset mnist --input_height=28 --output_height=28 --train
    $ python main.py --dataset celebA --input_height=108 --train --crop

To test with an existing model:

    $ python main.py --dataset mnist --input_height=28 --output_height=28
    $ python main.py --dataset celebA --input_height=108 --crop

Or, you can use your own dataset (without central crop) by:

    $ mkdir data/DATASET_NAME
    ... add images to data/DATASET_NAME ...
    $ python main.py --dataset DATASET_NAME --train
    $ python main.py --dataset DATASET_NAME
    $ # example
    $ python main.py --dataset=eyes --input_fname_pattern="*_cropped.png" --train

If your dataset is located in a different root directory:

    $ python main.py --dataset DATASET_NAME --data_dir DATASET_ROOT_DIR --train
    $ python main.py --dataset DATASET_NAME --data_dir DATASET_ROOT_DIR
    $ # example
    $ python main.py --dataset=eyes --data_dir ../datasets/ --input_fname_pattern="*_cropped.png" --train
    
    
    
    
    
    python main.py --dataset=TestImage --input_fname_pattern="*_traj.png" --input_height=288 --output_height=288 --train
    python main.py --dataset=TestImage --input_fname_pattern="*_traj.png" --input_height=288 --crop --train 
    python main.py --dataset=v2 --input_fname_pattern="*_traj.png" --input_height=512 --crop --train 
  python main.py --dataset=test --input_fname_pattern="*_traj.png" --input_height=288 --input_width=432 --output_height=288 --output_width=432  --train --crop 

python main.py --dataset=v5 --input_fname_pattern="*_traj.png" --input_height=64 --output_height=64  --train 

python main.py --dataset=v7 --input_fname_pattern="*_traj.png" --input_height=128 --output_height=128  --train 
python main.py --dataset=TestImage --input_fname_pattern="*_traj.png" --input_height=128 --output_height=128  --train 


