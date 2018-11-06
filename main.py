
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
import traceback
import numpy as np
import tensorflow as tf

from model import tfrecord_model_graph, average_gradients
from utils import tfrecord_dataset_reader, get_data, parser, get_part_name, change_partfile, Preprocess, ShowProcess
from tensorflow.python.client import device_lib

################################################################################
# # Load Configuration

parser = argparse.ArgumentParser()

parser.add_argument('--train_epochs', action="store",
                    dest="train_epochs", type=int, default=700,
                    help="Set maximum train epochs.")
parser.add_argument('--test_epochs', action="store",
                    dest="test_epochs", type=int, default=1,
                    help="Set maximum test epochs.")
parser.add_argument('--learning_rate', action="store", 
                    dest="lr", type=float, default=0.01,
                    help="Set learning rate.")
parser.add_argument('--train_batch_size', action="store", 
                    dest="train_batch_size", type=int, default=5000,
                    help="Set train mini-batch size")
parser.add_argument('--test_batch_size', action="store",
                    dest="test_batch_size", type=int, default=5000,
                    help="Set test mini-batch size")
parser.add_argument('--read_thread', action="store",
                    dest="num_thread", type=int, default=1,
                    help="Set number of threads to read data.")
parser.add_argument('--num_auc_threshold', action="store",
                    dest="auc_threshold", type=int, default=100,
                    help="Set number of thresholds when calc auc.")
parser.add_argument('--print_per_epochs', action="store", 
                    dest="print_per_epochs", type=int, default=1,
                    help="Set info print frequency during training.")
parser.add_argument('--save_per_epochs', action="store", 
                    dest="save_per_epochs", type=int, default=10,
                    help="Set model save frequency during training.")
parser.add_argument('--min_after_dequeue', action="store",
                    dest="min_after_dequeue", type=int, default=100,
                    help="The minimal number after dequeue.")
parser.add_argument('--pure_train_step', action="store",
                    dest="pure_train_step", type=int, default=100,
                    help="The pure train step before initializing test checking.")
parser.add_argument('--run_type', action="store",
                      dest="run_type", default="gpu",
                      help="Choose cpu or gpu.")
parser.add_argument('--num_workers', action="store",
                      dest="num_workers", default=1,type=int,
                      help="Set Number of workers.")
parser.add_argument('--num_cores', action="store",
                      dest="num_cores", default=1,type=int,
                      help="Set Number of cores.")
parser.add_argument('--worker_check_frequency', action="store",
                      dest="worker_check_frequency", default=1,type=int,
                      help="Set Number of cores.")
parser.add_argument('--min_allowed_loss_down', action="store",
                      dest="min_allowed_loss_down", default=0,type=float,
                      help="Set minimum allowed loss decrease value.")
parser.add_argument('--max_loss_increase', action="store",
                      dest="max_loss_increase", default=3,type=int,
                      help="Set maximum allowed test loss increase times contunuously.")
parser.add_argument('--local_data_dir', action="store",
                      dest="local_data_dir", default="./dataset",
                      help="Set local data directory.")
parser.add_argument('--test_local_data_dir', action="store",
                      dest="test_local_data_dir", default="./test_dataset",
                      help="Set local test data directory.")
parser.add_argument('--test_option', action="store",type=int,
                      dest="test_option", default=0,
                      help="Set whether to test model during training or not.")
parser.add_argument('--feature_dim', action="store",
                    dest="feature_dim", default=100,type=int,
                    help="Set feature dimemsions.")
parser.add_argument('--label_dim', action="store",
                    dest="label_dim", default=2,type=int,
                    help="Set label dimensions.")
parser.add_argument('--reload_model', action="store",
                    dest="reload_model", default="xxx",
                    help="Set path for reloading model checkpoint.")
parser.add_argument('--save_model_local_dir', action="store",
                    dest="save_model_local_dir", default="saved_models",
                    help="Set path for saving model local directory.")
parser.add_argument('--input_mode', action="store",
                    dest="input_mode", default="download",
                    help="Set input mode.")

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

################################################################################
# # Configure ps and worker
ps_hosts = args.ps_hosts.split(",")
worker_hosts = args.worker_hosts.split(",")

cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})
server = tf.train.Server(cluster, job_name=args.job_name, task_index=args.task_index)

# Check job_name: ps or worker
print(args.job_name)

if args.job_name == "ps":
    server.join()
elif args.job_name == "worker":
    with tf.device(tf.train.replica_device_setter(
        worker_device="/job:worker/task:%d" % args.task_index,
        cluster=cluster)):

        # # Check input mode
        if args.input_mode.lower() == "download":
            args.data_path = args.local_data_dir
            if args.test_option:
                test_data_address = args.test_local_data_dir
        elif args.input_mode.lower() == "placeholder":
            input_str = os.environ["INPUT_FILE_LIST"]
            # Here we get a string from os.environ
            # Use ast.literal_eval() to translate it into a dictionary
            input_dict = ast.literal_eval(input_str)
            args.data_path = [a for a in input_dict[args.local_data_dir]]
            if args.test_option:
                test_data_address = [a for a in input_dict[args.test_local_data_dir]]

        global_step = tf.Variable(0, name="global_step", trainable=False)
        
        # # Check Path
        if not os.path.exists(args.local_data_dir):
            os.mkdir(args.local_data_dir)
        if not os.path.exists(args.test_local_data_dir):
            os.mkdir(args.test_local_data_dir)
        if not os.path.exists(args.save_model_local_dir):
            os.mkdir(args.save_model_local_dir)

        # # Check Device
        print("Start Checking Device.")
        print(device_lib.list_local_devices())
        print(tf.__version__)
        print("Train Mini-batch size: "+str(args.train_batch_size))
        print("Test Mini-batch size: "+str(args.test_batch_size))

        # # Read Data
        batch_labels, batch_ids, batch_values, train_iterator = tfrecord_dataset_reader(args.data_path, \
                                                                  args.input_mode, args.train_epochs, args.train_batch_size)
        if args.test_option:
            test_batch_labels, test_batch_ids, \
                test_batch_values, test_iterator = tfrecord_dataset_reader(test_data_address, \
                                                     args.input_mode, args.test_epochs, args.test_batch_size)
        else:
            test_batch_labels, test_batch_ids, test_batch_values, test_iterator = None, None, None, None

        # # Define model and op
        with tf.device('/cpu:0'):
            tower_grads = []
            optimizer = tf.train.AdamOptimizer(learning_rate=args.lr)
            with tf.variable_scope(tf.get_variable_scope()):

                # Check run_type for gpu or cpu
                if args.run_type.lower() == "gpu":
                    device_range = range(args.num_cores)
                elif args.run_type.lower() == "cpu":
                    device_range = range(1, args.num_cores)
                    
                for i in device_range:
                    with tf.device('/'+args.run_type.lower()+':%d' % i):
                        if args.test_option:
                            W, b, cost, auc_update_op, test_cost, test_auc, test_auc_update_op, \
                            summary_op, saver = tfrecord_model_graph(batch_ids, batch_values, batch_labels, \
                                                                     test_batch_ids, test_batch_values, test_batch_labels, \
                                                                     args.feature_dim, args.label_dim, args.auc_threshold, \
                                                                     args.reload_model, args.test_option)
                        else:
                            W, b, cost, auc_update_op, \
                            summary_op, saver = tfrecord_model_graph(batch_ids, batch_values, batch_labels, \
                                                                     test_batch_ids, test_batch_values, test_batch_labels, \
                                                                     args.feature_dim, args.label_dim, args.auc_threshold, \
                                                                     args.reload_model, args.test_option)
                        tf.get_variable_scope().reuse_variables()
                        grads = optimizer.compute_gradients(cost)
                        tower_grads.append(grads)

            grads = average_gradients(tower_grads)
            apply_gradient_op = optimizer.apply_gradients(grads, global_step=global_step)
            train_op = apply_gradient_op
            init_global_op = tf.global_variables_initializer()
            init_local_op = tf.local_variables_initializer()
            writer = tf.summary.FileWriter(args.log_dir, graph=tf.get_default_graph())

        if args.test_option:
            # Save test loss calculation condition
            time_to_test = tf.Variable(0, name="time_to_test_list", trainable=False)
            time_to_exit = tf.Variable(0, name="time_to_exit", trainable=False)
            time_to_save = tf.Variable(0, name="time_to_save", trainable=False)
            test_loss_list = tf.Variable([0.0]*args.num_workers, name="test_loss_list", trainable=False)
            test_auc_list = tf.Variable([0.0]*args.num_workers, name="test_auc_list", trainable=False)
            test_state_list = tf.Variable([0]*args.num_workers, name="test_state_list", trainable=False)
            test_old_loss = tf.Variable(0.0, name="test_loss_old", trainable=False)
            test_loss_increase = tf.Variable(0, name="test_loss_increase", trainable=False)
            # Set update op
            time_to_test_update = tf.assign(time_to_test, 1)
            time_to_exit_update = tf.assign(time_to_exit, 1)
            time_to_save_update = tf.assign(time_to_save, 1)
            test_old_loss_update = tf.assign(test_old_loss,tf.reduce_sum(test_loss_list))
            test_loss_list_update = tf.assign(test_loss_list[args.task_index],test_loss_list[args.task_index]+test_cost/args.test_batch_size)
            test_auc_list_update = tf.assign(test_auc_list[args.task_index],test_auc_list[args.task_index]+test_auc)
            test_state_update = tf.assign(test_state_list[args.task_index], 1)
            test_loss_increase_update = tf.assign_add(test_loss_increase, 1)
            # Set reset op
            time_to_test_reset = tf.assign(time_to_test, 0)
            time_to_save_reset = tf.assign(time_to_save, 0)
            test_loss_reset = tf.assign(test_loss_list,[0.0]*args.num_workers)
            test_auc_reset = tf.assign(test_auc_list,[0.0]*args.num_workers)
            test_state_reset = tf.assign(test_state_list,[0]*args.num_workers)
            test_loss_increase_reset = tf.assign(test_loss_increase, 0)

        # For DEBUG
        #optimizer = tf.train.AdamOptimizer(learning_rate=args.lr)
        #cost, auc_update_op, summary_op, saver = tfrecord_model_graph(batch_ids, batch_values, batch_labels, args.feature_dim, args.label_dim, args.auc_threshold, args.reload_model, args.test_option)
        #train_op = optimizer.minimize(cost, global_step=global_step)
        #init_global_op = tf.global_variables_initializer()
        #init_local_op = tf.local_variables_initializer()
        #writer = tf.summary.FileWriter(args.log_dir, graph=tf.get_default_graph())

    # Config
    # For multiple gpu, pls set allow_soft_placement = True
    configuration = tf.ConfigProto(allow_soft_placement=True,
                        log_device_placement=False,
                        device_filters=["/job:ps", "/job:worker/task:%d" % args.task_index])

    scaffold = tf.train.Scaffold(local_init_op=tf.group(init_local_op,init_global_op,
                                            train_iterator.initializer,test_iterator.initializer),
                                 saver=saver,
                                 summary_op=summary_op
                                )
    
    # Get checkpoint directory
    if args.reload_model == "xxx":
        check_path = args.save_model_local_dir
    else:
        check_path = args.reload_model
    print("Check point dir: "+check_path)

    with tf.train.MonitoredTrainingSession(master=server.target, 
                                           is_chief=(args.task_index==0),
                                           scaffold=scaffold,
                                           config=configuration,
                                           checkpoint_dir=check_path,
                                           save_checkpoint_steps=args.save_per_epochs,
                                           save_summaries_steps=args.print_per_epochs
                                           ) as sess:

        print("Sesssion Created !")

        # Initialize
        sess.run(train_iterator.initializer)
        sess.run(test_iterator.initializer)
        sess.run(init_local_op)
        sess.run(init_global_op)

        # For checking parameters 
        #print("DEBUG")
        #print(W.eval(session=sess))
        #print(b.eval(session=sess))

        while True:
            try:
                # Check stop condition
                if sess.run(time_to_exit) == 1:
                    break

                start_time = time.time()
                step = sess.run(global_step)

                if step % args.save_per_epochs == 0:
                    sess.run(time_to_save_update)

                # # Train
                if (int(step % args.print_per_epochs) == 0 or sess.run(time_to_test) == 1) and step != 0 and step >= args.pure_train_step:
                    sess.run(time_to_test_update)
                    if args.test_option:
                        # Calc current train batch loss and batch auc
                        loss, auc = sess.run([cost, auc_update_op])
                        # Calc test set loss and auc
                        old_test_loss = sess.run(test_old_loss)
                        # Initialize test data iterator
                        sess.run(test_iterator.initializer)
                        test_iter_count = 0
                        while True:
                            try:
                                sess.run([test_loss_list_update, test_auc_update_op, test_auc_list_update])
                                test_iter_count += 1
                                # For debug
                                if test_iter_count % 1 == 0:
                                    print(test_iter_count)
                                    print(sess.run(test_loss_list))
                            except:
                                print("Testing: Except detected in task: "+str(args.task_index))
                                print(traceback.print_exc())
                                sess.run(test_iterator.initializer)
                                break

                        # Now this worker has finished test data calc
                        # Wait for other workers
                        sess.run(test_state_update)

                        while args.num_workers != sum(sess.run(test_state_list)):
                            print("Wait for another %d s."%args.worker_check_frequency)
                            time.sleep(args.worker_check_frequency)
                            print(sess.run(test_state_list))
                            if sess.run(time_to_exit) == 1:
                                break

                        # In case other worker cannot get correct state info
                        sess.run([test_state_update, time_to_test_reset])
                        time.sleep(2*args.worker_check_frequency)

                        # Specify process only for task 0
                        if args.task_index == 0:
                            # Now all workers have finished test data calc
                            current_test_loss = sum(sess.run(test_loss_list))
                            current_test_auc = sum(sess.run(test_auc_list))
                            # Check whether loss increase count reaches maximum allowed times or not
                            if old_test_loss != 0:
                                if current_test_loss > old_test_loss:
                                    increase_num = sess.run(test_loss_increase_update)
                                    if increase_num >= args.max_loss_increase:
                                        sess.run(time_to_exit_update)
                                        break
                                else:
                                    sess.run(test_loss_increase_reset)
                                    if old_test_loss - current_test_loss <= args.min_allowed_loss_down:
                                        break
                            # Reset all the test data loss and worker calc state
                            sess.run([test_old_loss_update, test_loss_reset, test_auc_reset, test_state_reset])
                            # Print related value
                            print("Epoch: %02d  train_loss: %.7f "
                                  "train_auc: %.5f test_loss: %.7f "
                                  "test_auc: %.5f" 
                                  % (step, float(loss)/args.train_batch_size, auc, \
                                  float(current_test_loss)/(args.num_workers * test_iter_count), \
                                  float(current_test_auc)/(args.num_workers * test_iter_count)))

                        # Run optimizer
                        sess.run(train_op)
                    else:
                        _, loss, auc = sess.run([train_op, cost, auc_update_op])
                        print("Epoch: %02d  " 
                              "train_loss: %.7f "
                              "train_auc: %.5f" 
                              % (step, float(loss)/args.train_batch_size, auc))

                    # Notify the value of step
                    print("Finished training for epoch "+str(step))

                    if args.run_type.lower() == "gpu":
                        print(os.system("nvidia-smi"))
                else:
                    # Just run optimizer
                    sess.run(train_op)
                    # Notify the value of step
                    print("Finished training for epoch "+str(step))

                # Check run time
                end_time = time.time()
                print('Train for epoch '+str(step)+': '+str(end_time-start_time)+' seconds.')
            except:
                print("Training: Exception detected in task: "+str(args.task_index))
                print(traceback.print_exc())
                sess.run(time_to_exit_update)
                continue
    print("Training Completed !")

