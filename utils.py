import os
import re
import sys
import time
import numpy as np
import tensorflow as tf

def tfrecord_dataset_reader(input_files, input_mode, epochs, batch_size):
    '''
    Set data input pipeline using tensorflow dataset api

    Args:
        input_files: (list) list all the input part file paths.
        input_mode: (string) "download" or placeholder
        epochs: (int) how many times you want to iterate your data.
        batch_size: (int) mini-batch size for dataset.
    '''
    assert len(input_files)>0, "Dataset reader receives empty input file list. Pls check your input setting."
    assert epochs>0, "Dataset iterate epochs must be > 0. Pls check your setting."

    # Generate files path list
    files = file_list_parser(input_files, input_mode)

    # tf.contrib.data.TFRecordDataset is available for tensorflow >= 1.2
    # For tensorflow >= 1.5, pls use tf.data.TFRecordDataset
    dataset = tf.data.TFRecordDataset(files)

    def _parser(example_proto):
        '''
        Parse example from tfrecord
        '''
        features_to_type = {"label": tf.FixedLenFeature([], tf.int64, default_value=0),
                    "ids": tf.VarLenFeature(tf.int64),
                    "vals": tf.VarLenFeature(tf.float32)}
        features = tf.parse_single_example(example_proto, features_to_type)

        return features["label"], features["ids"], features["vals"]

    # Put parsed record into tensor
    dataset = dataset.map(_parser)

    # Set repeat times
    dataset = dataset.repeat(count=epochs)

    # Set Mini-batch
    dataset = dataset.batch(batch_size)

    # Make one shot generator
    iterator = dataset.make_initializable_iterator()

    # Generate Data
    labels, ids, values = iterator.get_next()

    return labels, ids, values, iterator

def file_list_parser(data_dir, input_mode):
    '''
    Generate appropriate file list
    '''
    files = []
    if input_mode == "download" or input_mode.lower() == "download":
        if tf.gfile.IsDirectory(data_dir):
            print("DEBUG: file_queue_generator recieves a directory.")
            file_list = tf.gfile.ListDirectory(data_dir)
            print("DEBUG: Print File list")
            print(file_list)
            files = [os.path.join(data_dir, f) for f in file_list]
        elif data_dir.find(",") > 0:
            print("DEBUG: file_queue_generator recieves many directories.")
            files = tf.gfile.Glob(data_dir)
            dir_list = data_dir.split(",")
            for d in dir_list:
                file_list = tf.gfile.ListDirectory(d)
                files.extend = [os.path.join(d,f) for f in file_list]
    elif input_mode == "placeholder" or input_mode.lower() == "placeholder":
        files = data_dir
    return files


def libsvm_reader(file_queue, batch_size, min_after_dequeue, num_preprocess_threads):
    '''
    Read Data from prepared TFRecord file
    '''
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(file_queue)
    batch_serialized_examples = tf.train.shuffle_batch([serialized_example], batch_size=batch_size, capacity=min_after_dequeue + (num_preprocess_threads + 1) * batch_size, min_after_dequeue=min_after_dequeue, num_threads=num_preprocess_threads, allow_smaller_final_batch=True)
    feature_to_type = {"label": tf.FixedLenFeature([], tf.int64, default_value=0),
                       "ids": tf.VarLenFeature(tf.int64),
                       "vals": tf.VarLenFeature(tf.float32)}
    features = tf.parse_example(batch_serialized_examples, feature_to_type)

    return features["label"], features["ids"], features["vals"]

def file_queue_generator(data_dir, total_epochs, input_mode):
    '''
    Create File Queue
    '''
    files = []
    if input_mode == "download" or input_mode.lower() == "download":
        if tf.gfile.IsDirectory(data_dir):
            print("DEBUG: file_queue_generator recieves a directory.")
            file_list = tf.gfile.ListDirectory(data_dir)
            print("DEBUG: Print File list")
            print(file_list)
            files = [os.path.join(data_dir, f) for f in file_list]
        elif data_dir.find(",") > 0:
            print("DEBUG: file_queue_generator recieves many directories.")
            files = tf.gfile.Glob(data_dir)
            dir_list = data_dir.split(",")
            for d in dir_list:
                file_list = tf.gfile.ListDirectory(d)
                files.extend = [os.path.join(d,f) for f in file_list]
    elif input_mode == "placeholder" or input_mode.lower() == "placeholder":
        files = data_dir
    print("DEBUG file list: ")
    print(files)
    print("DEBUG: file list length = "+str(len(files)))
    filename_queue = tf.train.string_input_producer(files, total_epochs)
    print("filename_queue Finished !")
    return filename_queue

def get_data(input_file, total_epochs, batch_size, min_after_dequeue, thread_number, input_mode):
    '''
    Primary function for obtaining data
    '''
    filename_queue = file_queue_generator(input_file, total_epochs, input_mode)
    batch_labels, batch_ids, batch_values = libsvm_reader(filename_queue, batch_size=batch_size, min_after_dequeue=min_after_dequeue, num_preprocess_threads=thread_number)
    return batch_labels, batch_ids, batch_values

def parser(line, onehot=False):
    '''
    Params:
        line: non-empty string, one line input information from dataset.
        onehot: (Optional) Determine whether return onehot label or not.
    Return:
        features: numpy array of features.
        labels: numpy array of corresponded labels.
    '''
    assert isinstance(line,str), "Parser receives non-string input! Error!"
    assert len(line)>0, "Parser receives empty input string! Error!"
    
    # Read Info from original data
    line = re.sub(r'\s{2,}','',line)
    line = line.lstrip('\n').rstrip('\n').split()
    features = [a.split(':')[0] for a in line if ':' in a]
    labels = [line[0]]*len(features)
    
    # PreProcess features
    features = [(10-len(a))*'0'+a for a in features]
    features = [list(a) for a in features]
    features = np.array(features).astype(int)
    
    # PreProcess Labels
    if onehot:
        labels = [[1,0] if a=='0' else [0,1] for a in labels]
    else:
        labels = [int(a) for a in labels]
    labels = np.array(labels)
    
    return features, labels

def get_part_name(num):
    '''
    Get part-xxxxx name based on given number
    Params: 
        num: int
    Return:
        part-xxxxx: string
    '''
    assert isinstance(num,int), "get_part_name function receives non-int parameter. It should be int."
    assert num>0, "In get_part_name(num), num must be num>0."
    return "part-"+(5-len(str(num-1)))*'0'+str(num-1)

def change_partfile(num, address):
    '''
    Change to next partfile
    Params:
        num: int
        address: string
    Return:
        lines: list of string
    '''
    assert isinstance(num,int), "change_partfile function receives non-int parameter 'num'. It should be int."
    assert num>0, "In change_partfile(num,readline_batch), num must be num>0."
    assert isinstance(address, str) or isinstance(address, unicode), "change_partfile function receives non-string data address."
    assert len(address)>0, "change_partfile function receives empty data address! error!"

    if num>1:
        os.system("rm ./dataset/"+get_part_name(num-1))
    part_name = get_part_name(num)
    print("Downloading "+part_name)
    os.system("hdfs dfs -get "+address+part_name+" ./dataset/")
    f = open("./dataset/"+part_name,'r')
    lines = f.readlines()
    f.close()
    return lines

def Preprocess(split_ratio, lines):
    '''
    Split dataset into training data and test data.
    Then parse data to get processed features and labels.
    Params:
        split_ratio: int
        lines: list of string
    Return:
        x_train, x_test: numpy array, processed features
        y_train, y_test: numpy array, processed labels
    '''
    assert isinstance(split_ratio,float), "In Preprocess(split_ratio, lines), split_ratio should be float within (0,1)."
    assert split_ratio>0 and split_ratio<1, "In Preprocess(split_ratio, lines), split_ratio should be float within (0,1)."
    assert isinstance(lines,list), "Preprocess function receoves non-list lines! Error!"
    assert len(lines)>0, "Preprocess function receives empty lines! Error!"

    print("Start split dataset.")
    total_lines = len(lines)
    data_split_index = int(split_ratio * total_lines)
    train_lines = lines[:data_split_index]
    test_lines = lines[data_split_index:]
    print("There are "+str(total_lines)+" lines in this part file.")
    print("With split_ratio="+str(split_ratio)+", "+
            str(len(train_lines))+" lines are used for training set and "+
            str(len(test_lines))+" lines are used for testing set.")

    print("Start to parse train lines.")
    #process_bar_train = ShowProcess(len(train_lines), "Finish parsing train lines.")
    x_train_list = []
    y_train_list = []
    for line in train_lines:
        tmp_x_train, tmp_y_train = parser(line,onehot=True)
        #process_bar_train.show_process()
        x_train_list.append(tmp_x_train)
        y_train_list.append(tmp_y_train)
    x_train = np.vstack((x_train_list))
    y_train = np.vstack((y_train_list))
    x_train_list = []
    y_train_list = []
    
    print("Start to parse test lines.")
    #process_bar_test = ShowProcess(len(test_lines), "Finish parsing test lines.")
    x_test_list = []
    y_test_list = []
    for line in test_lines:
        tmp_x_test, tmp_y_test = parser(line,onehot=True)
        #process_bar_test.show_process()
        x_test_list.append(tmp_x_test)
        y_test_list.append(tmp_y_test)
    x_test = np.vstack((x_test_list))
    y_test = np.vstack((y_test_list))
    x_test_list = []
    y_test_list = []

    return x_train, y_train, x_test, y_test

class ShowProcess():
    '''
    Show Process in process bar
    '''
    i = 0 
    max_steps = 0 
    max_arrow = 50 
    infoDone = 'done'

    def __init__(self, max_steps, infoDone = 'Done'):
        self.max_steps = max_steps
        self.i = 0
        self.infoDone = infoDone

    def show_process(self, i=None):
        '''
        Start to show process.
        '''
        if i is not None:
            self.i = i
        else:
            self.i += 1
        num_arrow = int(self.i * self.max_arrow / self.max_steps)
        num_line = self.max_arrow - num_arrow 
        percent = self.i * 100.0 / self.max_steps 
        process_bar = '[' + '>' * num_arrow + '-' * num_line + ']'\
                      + '%.2f' % percent + '%' + '\r' 
        sys.stdout.write(process_bar) 
        sys.stdout.flush()
        if self.i >= self.max_steps:
            self.close()

    def close(self):
        '''
        Proces bar end.
        '''
        print('')
        print(self.infoDone)
        self.i = 0

