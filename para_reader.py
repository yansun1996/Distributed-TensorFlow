
# coding: utf-8

# # Import

import os
import re
import ast
import sys
import copy
import time
import argparse
import commands
import numpy as np
import tensorflow as tf

from model import tfrecord_model_graph, average_gradients
from utils import get_data, parser, get_part_name, change_partfile, Preprocess, ShowProcess
from tensorflow.python.client import device_lib

################################################################################
# # Load Configuration

parser = argparse.ArgumentParser()

parser.add_argument('--input_file', action="store",
                    dest="input_file", default="xxx",
                    help="Set input file.")
parser.add_argument('--output_file', action="store",
                    dest="output_file", default="xxx",
                    help="Set output file.")
parser.add_argument('--meta_graph', action="store",
                    dest="meta_graph", default="",
                    help="Set meta graph path.")
parser.add_argument('--check_point', action="store",
                    dest="check_point", default="",
                    help="Set checkpoint path.")

################################################################################
# # Start sina ML platform required arg

parser.add_argument('--log_dir', action="store", 
                    dest="log_dir", default="./logs/",
                    help="Set logs save path. Sina ML platform required arg.")
parser.add_argument('--train_dir', action="store", 
                    dest="model_path", default="",
                    help="Set model save path. Sina ML platform required arg.")
parser.add_argument('--data_dir', action="store", dest="data_path", 
                    default="",
                    help="Set remote data path. Sina ML platform required arg.")
parser.add_argument('--ps_hosts', action="store",
                      dest="ps_hosts", default="./logs/",
                      help="Set logs save path. Sina ML platform required arg.")
parser.add_argument('--worker_hosts', action="store",
                      dest="worker_hosts", default="./logs/",
                      help="Set logs save path. Sina ML platform required arg.")
parser.add_argument('--job_name', action="store",
                      dest="job_name", default="worker",
                      help="Set logs save path. Sina ML platform required arg.")
parser.add_argument('--task_index', action="store",
                      dest="task_index", default=0,type=int,
                      help="Set logs save path. Sina ML platform required arg.")

# # End sina ML required arg
################################################################################

args = parser.parse_args(sys.argv[1:])

sess = tf.Session()

saver = tf.train.import_meta_graph(args.meta_graph, clear_devices=True)
saver.restore(sess, args.check_point)

weights = sess.run(tf.get_default_graph().get_tensor_by_name("logistic_regression/weights:0"))
b = sess.run(tf.get_default_graph().get_tensor_by_name("logistic_regression/bias:0"))


outputfile = open(args.output_file,'w')

with open(args.input_file, 'r') as f:
    for line in f:
        tmp = line.strip('\n')
        key = tmp.split(' ')[2]
        outputfile.write(tmp + ' ' + str(weights[int(key)-1][0]) + '\n' )

outputfile.write(str(b) + '\n')
outputfile.close()
sess.close()


