Tensorflow code for distributed computing on large dataset. The distributed training platform is similar to [XLearning](https://github.com/Qihoo360/XLearning).

# File Description

* ```./dataset``` : Folder for saving data file

* ```./logs``` : Folder for saving tensorboard logs

* ```./saved_models``` : Folder for saving Tensorflow models

* ```main.py``` : Main

* ```models.py``` : Define model graph in this script.

* ```utils.py``` : Useful functions enclosed in this script.

* ```clean.sh``` : clean pyc file and temp data file in this repository

* ```submit.sh``` : Submit this repository to Sina ML platform to initialize distributed learning task.

# How to run

1. Determine your parameters.

4. Run ```sh submit.sh```

5. Check results on the hadoop UI address (found in the output) in your browser.

# Sample Submit Command

```
$ML_HOME/bin/ml-submit \
  --app-type "tensorflow"\
  --app-name "tensorflow_lr_optimal_method"   \
  --cacheArchive <Address for your packaged Python environment>#Python \
  --files main.py,model.py,utils.py  \
  --board-enable true\
  --boardHistoryDir <Address for your TensorBoard output directory on HDFS> \
  --input-strategy DOWNLOAD \
  --output-strategy UPLOAD\
  --input <Address for your input train data>#dataset \
  --input <Address for your input test data>#test_dataset \
  --output <Address for your saved models>#saved_models \
  --worker-memory 16G\
  --worker-cores 1 \
  --worker-num 100 \
  --worker-gpu-cores 0 \
  --ps-num 1\
  --ps-cores 1\
  --ps-memory 16G \
  --launch-cmd "Python/bin/python main.py 
                --train_epochs 10000
                --test_epochs 1
                --train_batch_size 100000
                --test_batch_size 100000
                --feature_dim 100 
                --label_dim 1
                --learning_rate 0.01
                --print_per_epochs 100 
                --save_per_epochs 10
                --reload_model xxx
                --run_type cpu
                --num_workers 100
                --num_cores 4
                --worker_check_frequency 1
                --max_loss_increase 2
                --input_mode download 
                --save_model_local_dir saved_models 
                --local_data_dir dataset 
                --test_local_data_dir test_dataset 
                --test_option 1" 

```

# Submit Tutorial

The distributed training platform this repository relies on is similar to [XLearning](https://github.com/Qihoo360/XLearning). You could refer to that repository for how to submit.
