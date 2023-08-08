import os
import scipy.misc
import numpy as np

from newmodel import DCGAN
from utils import pp, visualize, to_json, show_all_variables

import tensorflow as tf

flags = tf.app.flags
flags.DEFINE_integer("epoch", 25, "Epoch to train [25]")
flags.DEFINE_float("learning_rate", 0.0002, "Learning rate of for adam [0.0002]")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
flags.DEFINE_float("train_size", np.inf, "The size of train images [np.inf]")
flags.DEFINE_integer("batch_size", 64, "The size of batch images [64]")
flags.DEFINE_integer("input_height", 108, "The size of image to use (will be center cropped). [108]")
flags.DEFINE_integer("input_width", None,
                     "The size of image to use (will be center cropped). If None, same value as input_height [None]")
flags.DEFINE_integer("output_height", 64, "The size of the output images to produce [64]")
flags.DEFINE_integer("output_width", None,
                     "The size of the output images to produce. If None, same value as output_height [None]")
flags.DEFINE_string("dataset", "celebA", "The name of dataset [celebA, mnist, lsun]")
flags.DEFINE_string("input_fname_pattern", "*.jpg", "Glob pattern of filename of input images [*]")
# flags.DEFINE_string("checkpoint_dir", "/home/beakash/ntu/caochu/MQ_test2/checkpoint", "Directory name to save the checkpoints [checkpoint]")
# flags.DEFINE_string("data_dir", "/home/beakash/ntu/caochu/MQ_test2/data", "Root directory of dataset [data]")
# flags.DEFINE_string("sample_dir", "/home/beakash/ntu/caochu/MQ_test2/samples", "Directory name to save the image samples [samples]")

flags.DEFINE_string("checkpoint_dir", "/home/beakash/Thesis/KDD2021_guizu/DCGAN Model/checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("data_dir", "/home/beakash/Thesis/KDD2021_guizu/DCGAN Model/porto", "Root directory of dataset [data]")
flags.DEFINE_string("sample_dir", "/home/beakash/Thesis/KDD2021_guizu/DCGAN Model/Sample", "Directory name to save the image samples [samples]")
flags.DEFINE_boolean("train", False, "True for training, False for testing [False]")
flags.DEFINE_boolean("crop", False, "True for training, False for testing [False]")
flags.DEFINE_boolean("visualize", False, "True for visualizing, False for nothing [False]")
flags.DEFINE_integer("generate_test_images", 100, "Number of images to generate during test. [100]")
FLAGS = flags.FLAGS


def main(_):
    pp.pprint(flags.FLAGS.__flags)
    run_config=tf.ConfigProto()
    run_config.gpu_options.allow_growth=True
    print("Hello this is main")
    data_X, data_y = load_mnist()
    # with tf.Session(config=run_config) as sess:
    #     dcgan=DCGAN



if __name__ == '__main__':
    tf.app.run()
