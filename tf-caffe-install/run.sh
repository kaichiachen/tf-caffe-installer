#!/bin/bash
export P1=$1
function menu {
    echo -e 'What do you want to do?'
    echo -e '1. Install Environment'
    echo -e '2. Run in SSD300 X 300'
    echo -e '3. Train in SSD300 X 300'
    echo -en '\t\nEnter option: '
    read -n 1 option
}

function install_dev {

    # install dependancy package
    sudo apt-get -y update
    sudo apt-get -y upgrade
    sudo apt-get -y dist-upgrade
    sudo apt-get -y autoremove
    sudo apt-get -y install git
    sudo apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        git \
        wget \
        curl \
        unzip \
        libatlas-base-dev \
        libboost-all-dev \
        libgflags-dev \
        libgoogle-glog-dev \
        libhdf5-serial-dev \
        libleveldb-dev \
        liblmdb-dev \
        libopencv-dev \
        libprotobuf-dev \
        libsnappy-dev \
        protobuf-compiler \
        python-dev \
        python-numpy \
        python-pip \
        python-setuptools \
        python-scipy

    # upgrade pip version
    sudo pip install --upgrade pip
}

function install_tensorflow {
    cd $MYHOME

    if [ $P1 = 'cpu' ]
    then
        sudo pip install tensorflow==1.4
    else
        sudo pip install tensorflow-gpu==1.4
    fi

    # check object_detection official model
    if [ ! -d "./models" ]
    then
        git clone https://github.com/tensorflow/models.git

        # install dependancy package
        sudo apt-get -y install python-tk
        sudo apt-get -y install python-pil python-lxml
        sudo pip install pillow
        sudo pip install matplotlib
        sudo pip install lxml

        # set environment variable
        export TENSORFLOW_ROOT=$MYHOME/models
        cd $TENSORFLOW_ROOT/research
        protoc object_detection/protos/*.proto --python_out=.
        export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim

        # run tensorflow installation checker
        python object_detection/builders/model_builder_test.py

        # make directory for train
        mkdir -p $TENSORFLOW_ROOT/research/object_detection/models/model/train/
    fi
}

function install_caffe {
    cd $MYHOME

    # check caffe official model
    if [ ! -d "./caffe" ]
    then
        git clone https://github.com/weiliu89/caffe.git --branch ssd
        export CAFFE_ROOT=$MYHOME/caffe
        export PYTHONPATH=$MYHOME/caffe/python
        cd $CAFFE_ROOT/python

        # install dependancy package
        for req in $(cat requirements.txt); do sudo pip install $req; done
        sudo pip install pillow --upgrade

        cd $CAFFE_ROOT

        # compile caffe
        mkdir build && cd build
        if [ $P1 = 'cpu' ]
        then
            cmake -DCPU_ONLY=1 ..
            make -j"$(nproc)"
        else
            cmake -DUSE_CUDNN=1 -DUSE_NCCL=1 ..
            make -j"$(nproc)"
        fi
        make pycaffe
    fi
    cd $CAFFE_ROOT
}

function run_ssd_caffe_video {
    cd $MYHOME
    export CAFFE_ROOT=$MYHOME/caffe
    export PYTHONPATH=$CAFFE_ROOT/python

    # check caffe official model
    if [ ! -d "./caffe" ]
    then
        install_caffe
    fi

    if [ ! -d "./caffe/models/VGGNet/VOC0712" ]
    then
        cp ../models_VGGNet_VOC0712_SSD_300x300.tar.gz ./
        tar -xvf models_VGGNet_VOC0712_SSD_300x300.tar.gz
    fi

    cd $MYHOME
    cp ./video1.mp4 $CAFFE_ROOT/examples/videos/ILSVRC2015_train_00755001.mp4
    cp ./ssd_pascal_video.py $CAFFE_ROOT/examples/ssd/ssd_pascal_video.py

    cd $CAFFE_ROOT
    python examples/ssd/ssd_pascal_video.py
}

function run_ssd_caffe {
    cd $MYHOME
    export CAFFE_ROOT=$MYHOME/caffe
    export PYTHONPATH=$CAFFE_ROOT/python

    # check caffe official model
    if [ ! -d "./caffe" ]
    then
        install_caffe
    fi

    cd $MYHOME
    # check our test images
    if [ ! -d "./frames" ]
    then
        tar -xvf output.tar
    fi
    cd $CAFFE_ROOT
    cp ../my_ssd_detect.py ./examples/ssd

    if [ ! -d "./caffe/models/VGGNet/VOC0712" ]
    then
        cp ../models_VGGNet_VOC0712_SSD_300x300.tar.gz ./
        tar -xvf models_VGGNet_VOC0712_SSD_300x300.tar.gz
    fi

    rm -r $MYHOME/caffe_detect_output
    mkdir $MYHOME/caffe_detect_output

    # run pre-train model to detect our test images
    if [ $P1 = 'cpu' ]
    then
        python examples/ssd/my_ssd_detect.py cpu
    else
        python examples/ssd/my_ssd_detect.py gpu
    fi
}

function run_ssd_tensorflow {
    cd $MYHOME
    # check object_detection official model
    if [ ! -d "./models" ]
    then
        install_tensorflow
    fi

    cd $MYHOME
    # check our test images
    if [ ! -d "./frames" ]
    then
        tar -xvf output.tar
    fi

    # set environment variable
    export TENSORFLOW_ROOT=$MYHOME/models
    cd $TENSORFLOW_ROOT/research
    protoc object_detection/protos/*.proto --python_out=.
    export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim


    # download pre-train model
    cd $TENSORFLOW_ROOT/research/object_detection/models/model
    if [ ! -d "./$1" ]
    then
        wget http://download.tensorflow.org/models/object_detection/$1.tar.gz
        tar -xzvf $1.tar.gz
    fi
    rm -r $MYHOME/tf_detect_output
    mkdir $MYHOME/tf_detect_output


    sed -e "s/ssd_inception_v2_coco_2017_11_17/$1/;" $MYHOME/object_detection_sample.py > $TENSORFLOW_ROOT/research/object_detection/object_detection_sample.py
    # run pre-train model to detect our test images
    python $TENSORFLOW_ROOT/research/object_detection/object_detection_sample.py
}

function train_ssd_caffe {
    cd $MYHOME
    # check caffe official model
    if [ ! -d "./caffe" ]
    then
        install_caffe
    fi
    cd $MYHOME
    # check train data
    if [ ! -d "./data/VOCdevkit" ]
    then
        wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
        tar -xvf VOCtrainval_06-Nov-2007.tar
        rm VOCtrainval_06-Nov-2007.tar

        wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
        tar -xvf VOCtest_06-Nov-2007.tar
        rm VOCtest_06-Nov-2007.tar

        mkdir -p $MYHOME/data
        cp -r ./VOCdevkit $MYHOME/data/
    fi

    # set environment variable
    export CAFFE_ROOT=$MYHOME/caffe
    export PYTHONPATH=$CAFFE_ROOT/python

    cd $CAFFE_ROOT
    cp ../my_ssd_detect.py ./examples/ssd

    # check pre-train model download or not
    if [ ! -d "./caffe/models/VGGNet/VOC0712" ]
    then
        cp ../models_VGGNet_VOC0712_SSD_300x300.tar.gz ./
        tar -xvf models_VGGNet_VOC0712_SSD_300x300.tar.gz
    fi

    # change privilege of shell script
    chmod +x ./data/VOC0712/create_list.sh
    chmod +x ./data/VOC0712/create_data.sh
    
    # Create the trainval.txt, test.txt, and test_name_size.txt in data/VOC0712/
    ./data/VOC0712/create_list.sh
    # You can modify the parameters in create_data.sh if needed.
    # It will create lmdb files for trainval and test with encoded original image:
    #   - $HOME/data/VOCdevkit/VOC0712/lmdb/VOC0712_trainval_lmdb
    #   - $HOME/data/VOCdevkit/VOC0712/lmdb/VOC0712_test_lmdb
    # and make soft links at examples/VOC0712/
    ./data/VOC0712/create_data.sh

    if [ $P1 = 'cpu' ]
    then
        sed 's/gpus = "0,1,2,3"/gpus = ""/; 
         s/if num_gpus > 0:/if num_gpus > 0 and len(gpus) > 0:/; 
         s/VGG_ILSVRC_16_layers_fc_reduced.caffemodel/VOC0712\/SSD_300x300\/VGG_VOC0712_SSD_300x300_iter_120000.caffemodel/; ' $CAFFE_ROOT/examples/ssd/ssd_pascal.py > $CAFFE_ROOT/examples/ssd/ssd_pascal.py.tmp
    else
        sed 's/gpus = "0,1,2,3"/gpus = "0"/; 
         s/if num_gpus > 0:/if num_gpus > 0 and len(gpus) > 0:/; 
         s/VGG_ILSVRC_16_layers_fc_reduced.caffemodel/VOC0712\/SSD_300x300\/VGG_VOC0712_SSD_300x300_iter_120000.caffemodel/;' $CAFFE_ROOT/examples/ssd/ssd_pascal.py > $CAFFE_ROOT/examples/ssd/ssd_pascal.py.tmp
    fi
    
    mv $CAFFE_ROOT/examples/ssd/ssd_pascal.py.tmp $CAFFE_ROOT/examples/ssd/ssd_pascal.py

    # It will create model definition files and save snapshot models in:
    #   - $CAFFE_ROOT/models/VGGNet/VOC0712/SSD_300x300/
    # and job file, log file, and the python script in:
    #   - $CAFFE_ROOT/jobs/VGGNet/VOC0712/SSD_300x300/
    # and save temporary evaluation results in:
    #   - $HOME/data/VOCdevkit/results/VOC2007/SSD_300x300/
    # It should reach 77.* mAP at 120k iterations.
    if [ $P1 = 'cpu' ]
    then
        python examples/ssd/ssd_pascal.py cpu
    else
        python examples/ssd/ssd_pascal.py gpu
    fi

}

function train_ssd_tensorflow {
    cd $MYHOME

    # check object_detection official model
    if [ ! -d "./models" ]
    then
        install_tensorflow
    fi

    cd $MYHOME
    # check train data
    if [ ! -d "./VOCdevkit" ]
    then
        wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
        tar -xvf VOCtrainval_06-Nov-2007.tar
        rm VOCtrainval_06-Nov-2007.tar

        wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
        tar -xvf VOCtest_06-Nov-2007.tar
        rm VOCtest_06-Nov-2007.tar
    fi

    export TENSORFLOW_ROOT=$MYHOME/models
    # download pre-train model
    cd $TENSORFLOW_ROOT/research/object_detection/models/model
    if [ ! -d "./$1" ]
    then
        wget http://download.tensorflow.org/models/object_detection/$1.tar.gz
        tar -xzvf $1.tar.gz
    fi

    # set environment variable
    cd $TENSORFLOW_ROOT/research
    protoc object_detection/protos/*.proto --python_out=.
    export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim

    # convert data to tfrecord format
    python $TENSORFLOW_ROOT/research/object_detection/dataset_tools/create_pascal_tf_record.py \
            --label_map_path=$TENSORFLOW_ROOT/research/object_detection/data/pascal_label_map.pbtxt \
            --data_dir=$MYHOME/VOCdevkit/ \
            --year=VOC2007 \
            --set=train \
            --output_path=$TENSORFLOW_ROOT/research/object_detection/models/model/mscoco_train.record
    
    # edit train config
    cp $TENSORFLOW_ROOT/research/object_detection/data/pascal_label_map.pbtxt $TENSORFLOW_ROOT/research/object_detection/models/model/mscoco_label_map.pbtxt
    
    if [[ $1 = *"ssd"* ]]
    then
        sed -e "s/PATH_TO_BE_CONFIGURED/\/home\/ubuntu\/models\/research\/object_detection\/models\/model/; 
            s/model.ckpt/$1\/model.ckpt/;" $TENSORFLOW_ROOT/research/object_detection/samples/configs/ssd_inception_v2_coco.config > $TENSORFLOW_ROOT/research/object_detection/models/model/pipeline.config
    else
        sed -e "s/PATH_TO_BE_CONFIGURED/\/home\/ubuntu\/models\/research\/object_detection\/models\/model/; 
            s/model.ckpt/$1\/model.ckpt/;" $TENSORFLOW_ROOT/research/object_detection/samples/configs/faster_rcnn_inception_resnet_v2_atrous_coco.config > $TENSORFLOW_ROOT/research/object_detection/models/model/pipeline.config
    fi
    cd $TENSORFLOW_ROOT/research
    
    # train the model
    python object_detection/train.py \
        --logtostderr \
        --pipeline_config_path $TENSORFLOW_ROOT/research/object_detection/models/model/pipeline.config \
        --train_dir $TENSORFLOW_ROOT/research/object_detection/models/model/train
}

export MYHOME=/home/ubuntu
menu
echo
case $option in
0)
    install_dev ;;
1)
    echo -e 'Which one you want to install?'
    echo -e '1. tensorflow'
    echo -e '2. caffe'
    echo -e '3. tensorflow & caffe'
    echo -en '\t\nEnter option: '
    read -n 1 option
    echo
    case $option in   
    1)
        install_dev
        cd $MYHOME
        rm -r models
        install_tensorflow ;;
    2)
        install_dev
        cd $MYHOME
        rm -r caffe
        install_caffe ;;
    3)
        install_dev
        cd $MYHOME
        rm -r models
        rm -r caffe
        install_tensorflow
        install_caffe ;;
    *)
        echo 'leave' ;;
    esac ;;

2)
    echo -e 'Which one platform you want to run?'
    echo -e '1. tensorflow'
    echo -e '2. caffe'
    echo -en '\t\nEnter option: '
    read -n 1 option
    echo
    case $option in   
    1)
        echo -e 'Which one model you want to run?'
        echo -e '1. faster-rcnn'
        echo -e '2. ssd'
        echo -en '\t\nEnter option: '
        read -n 1 option
        echo
        case $option in   
        1)
            run_ssd_tensorflow 'faster_rcnn_inception_v2_coco_2017_11_08';;
        2)
            run_ssd_tensorflow 'ssd_inception_v2_coco_2017_11_17';;
        *)
            echo 'leave' ;;
        esac ;;
    2)
        echo -e 'Which input type do you want to run?'
        echo -e '1. picture'
        echo -e '2. video'
        echo -en '\t\nEnter option: '
        read -n 1 option
        echo
        case $option in   
        1)
            run_ssd_caffe ;;
        2)
            run_ssd_caffe_video ;;
        *)
            echo 'leave' ;;
        esac ;;
    *)
        echo 'leave' ;;
    esac ;;
3)
    echo -e 'Which one platform you want to train?'
    echo -e '1. tensorflow'
    echo -e '2. caffe'
    echo -en '\t\nEnter option: '
    read -n 1 option
    echo
    case $option in   
    1)
        echo -e 'Which one model you want to train?'
        echo -e '1. faster-rcnn'
        echo -e '2. ssd'
        echo -en '\t\nEnter option: '
        read -n 1 option
        echo
        case $option in   
        1)
            train_ssd_tensorflow 'faster_rcnn_inception_v2_coco_2017_11_08';;
        2)
            train_ssd_tensorflow 'ssd_inception_v2_coco_2017_11_17';;
        *)
            echo 'leave' ;;
        esac ;;
    2)
        train_ssd_caffe ;;
    *)
        echo 'leave' ;;
    esac ;;
*)
    echo 'leave' ;;
esac