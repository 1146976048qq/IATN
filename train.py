import tensorflow as tf
from utils import get_data_info, read_data, load_word_embeddings
from model import IATN
from evals import *
import os
import time
import math

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '4'

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
tf.enable_eager_execution(config=config)
