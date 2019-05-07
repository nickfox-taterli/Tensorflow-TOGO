# 如何在Windows 10上使用TensorFlow-GPU生成训练数据

## 步骤
### 1. 安装 TensorFlow-GPU 1.13.1

安装过程相当的简单,只要注意下载对应版本的附加组件就可以,注意,不能是更新或者更旧的组件,唯一版本.

 1. Python 3.6 <= 加入到PATH
 2. Anaconda3 5.2.0 <= 安装时提示的两个勾都要勾上,加入到PATH,会有警告,可无视.
 3. CUDA v10.0 <= 完全安装.
 4. cuDNN v7.5.1.10 for CUDA v10.0 <= 解压到D:\cuda了,如果你用别的路径,后续要修改.
 5. VS2015 (VC14) 代码生成工具 <= 不能用VS2019/VS2017代替.

接下来的事情比较简单,把(系统)环境变量PATH配置好.

 - D:\cuda\bin
 - D:\cuda\include
 - D:\cuda\lib
 - C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.0\lib
 - C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.0\include
 - C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.0\extras\CUPTI\lib64
 - C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.0\libnvvp
 - C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.0\bin

### 2. 设置Anaconda虚拟环境
#### 2a. 下载模型
下载 https://github.com/tensorflow/models 的最新版本,你可以通过点击"Clone or Download"下载ZIP文件,然后解压,我目前解压到D:\tf\models中.

#### 2b. 下载Faster-RCNN-Inception-V2-COCO基础模型
TensorFlow 提供很多基础模型给大家下载,具体猛戳 => [model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md). 一些模型注重精度,一些模型注重速度,具体看需求选择.精度高的训练速度也会变得非常慢.租GPU云的我就不说了.

我这里实验具体下载的是 [faster_rcnn_inception_v2_coco](http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_v2_coco_2018_01_28.tar.gz),如果你是纯新手,建议先跟着我来操作一遍.

#### 2c. 创建Anaconda虚拟环境
这里创建一个叫tensorflow1的虚拟环境.
```
C:\> conda create -n tensorflow1 pip python=3.6
```
激活这个虚拟环境:
```
C:\> activate tensorflow1
```
安装GPU支持:
```
(tensorflow1) C:\> pip install --ignore-installed --upgrade tensorflow-gpu
```
安装其他支持:
```
(tensorflow1) C:\> conda install -c anaconda protobuf
(tensorflow1) C:\> pip install pillow
(tensorflow1) C:\> pip install lxml
(tensorflow1) C:\> pip install Cython
(tensorflow1) C:\> pip install jupyter
(tensorflow1) C:\> pip install matplotlib
(tensorflow1) C:\> pip install pandas
(tensorflow1) C:\> pip install opencv-python
(tensorflow1) C:\> pip install pycocotools <= 通常会失败,谷歌能找到方法.
```

#### 2d. 配置 PYTHONPATH 环境变量
他必须包含\models,\models\research,\models\research\slim目录:
```
(tensorflow1) C:\> set PYTHONPATH=D:\tf\models;D:\tf\models\research;D:\tf\models\research\slim
```
(注意: 每次退出虚拟环境,都要重新配置此环境变量,不推荐设置为系统环境变量,因为有些库与系统Python库是重名的.)

#### 2e. 编译Protobufs 并运行 setup.py
Tensorflow要用Protobuf作为训练的参数,正常来说,运行protoc并指定通配符输出就可以了,但是在Windows上不起作用,所以只能把每个文件写一下了.

切换到 \models\research 目录并执行:
```
protoc --python_out=. .\object_detection\protos\anchor_generator.proto .\object_detection\protos\argmax_matcher.proto .\object_detection\protos\bipartite_matcher.proto .\object_detection\protos\box_coder.proto .\object_detection\protos\box_predictor.proto .\object_detection\protos\eval.proto .\object_detection\protos\faster_rcnn.proto .\object_detection\protos\faster_rcnn_box_coder.proto .\object_detection\protos\grid_anchor_generator.proto .\object_detection\protos\hyperparams.proto .\object_detection\protos\image_resizer.proto .\object_detection\protos\input_reader.proto .\object_detection\protos\losses.proto .\object_detection\protos\matcher.proto .\object_detection\protos\mean_stddev_box_coder.proto .\object_detection\protos\model.proto .\object_detection\protos\optimizer.proto .\object_detection\protos\pipeline.proto .\object_detection\protos\post_processing.proto .\object_detection\protos\preprocessor.proto .\object_detection\protos\region_similarity_calculator.proto .\object_detection\protos\square_box_coder.proto .\object_detection\protos\ssd.proto .\object_detection\protos\ssd_anchor_generator.proto .\object_detection\protos\string_int_label_map.proto .\object_detection\protos\train.proto .\object_detection\protos\keypoint_box_coder.proto .\object_detection\protos\multiscale_anchor_generator.proto .\object_detection\protos\graph_rewriter.proto .\object_detection\protos\calibration.proto
```
上面的命令主要是创建每个proto代表的pb2文件.

然后用安装脚本来安装:
```
(tensorflow1) D:\tf\models\research> python setup.py build
(tensorflow1) D:\tf\models\research> python setup.py install
```

#### 2f. 测试工作状态
切换到object_detection目录下,运行jupyter测试脚本:
```
(tensorflow1)  D:\tf\models\research\object_detection> jupyter notebook object_detection_tutorial.ipynb
```
一步一步执行,最终会看到输出一个测试图,那么,就是测试成功了,否则,检测其他环境配置.
![TestImg](https://www.lijingquan.net/wp-content/uploads/2019/05/img_5cd17e9dda5f2.png)

### 3. 标记图片
#### 3a. 准备图片
准备一大堆图片(自然拍照就可以),越多越好,至少也要几百张.并且分辨率不能太低(小于32*32),也不能太高(大于720*1280),文件大小也不能太大(小于200K),文件大小太大,分辨率太高的话,会严重影响训练速度,分辨率太小的话,影响准确度.

一般我们随机防止20%的图片到测试集 \object_detection\images\test directory, 还有 80% 数据用于训练集 \object_detection\images\train directory.如果数据量超大(大于百万图),还应该有验证集.

#### 3b. 标记图片
标记工具:[LabelImg](https://github.com/tzutalin/labelImg)

![LabelImg](https://www.lijingquan.net/wp-content/uploads/2019/05/img_5cd17dd5b717e.png)

具体如何标记的,就是拿框框框选,然后保存,然后继续下张图~ 如果不懂,那么可以自己谷歌一下.

### 4. Generate Training Data
当你标记完成之后,还应该把每个XML数据,转成单一的CSV数据:
```
(tensorflow1) D:\tf\models\research\object_detection> python xml_to_csv.py
```
然后还要写一下标记的指导文件generate_tfrecord.py:
```
def class_text_to_int(row_label):
    if row_label == 'nine':
        return 1
    elif row_label == 'ten':
        return 2
    elif row_label == 'jack':
        return 3
    elif row_label == 'queen':
        return 4
    elif row_label == 'king':
        return 5
    elif row_label == 'ace':
        return 6
    else:
        return None
```
最后,便可生成训练记录.(在object_detection目录执行):
```
python generate_tfrecord.py --csv_input=images\train_labels.csv --image_dir=images\train --output_path=train.record
python generate_tfrecord.py --csv_input=images\test_labels.csv --image_dir=images\test --output_path=test.record
```

### 5. 创建映射文件
#### 5a. 映射表
映射文件具体后缀是pbtxt,实际上他还是txt,不过注意你保存的是Unix格式的内容,举个例子,我的pbtxt内容如下.我储存到training/labelmap.pbtxt
```
item {
  id: 1
  name: 'nine'
}

item {
  id: 2
  name: 'ten'
}

item {
  id: 3
  name: 'jack'
}

item {
  id: 4
  name: 'queen'
}

item {
  id: 5
  name: 'king'
}

item {
  id: 6
  name: 'ace'
}
```

#### 5b. 配置训练
复制D:\tf\models\research\object_detection\samples\configs\faster_rcnn_inception_v2_pets.config 到 \object_detection\training 目录然后编辑.

- Line 9. num_classes <= 训练集里类型数,有多少就写多少,我这里写36.
- Line 106. fine_tune_checkpoint <= "D:/tf/models/research/object_detection/faster_rcnn_inception_v2_coco_2018_01_28/model.ckpt"

- Lines 123 and 125. :
  - input_path : "D:/tf/models/research/object_detection/train.record"
  - label_map_path:  "D:/tf/models/research/object_detection/training/labelmap.pbtxt"

- Line 130. num_examples <= \images\test 目录有多少图片就写多少,我这里有67张.

- Lines 135 and 137. :
  - input_path : "D:/tf/models/research/object_detection/test.record"
  - label_map_path:  "D:/tf/models/research/object_detection/training/labelmap.pbtxt"

### 6. 开始训练
训练就简单了,切换到research目录下,进行训练:
```
python object_detection\model_main.py --pipeline_config_path=object_detection\training\faster_rcnn_inception_v2_pets.config --model_dir=object_detection\images --num_train_steps=50000 --sample_1_of_n_eval_examples=1 --alsologtostderr
```
你可以打开另一个Anaconda,设置好PYTHONPATH,进入虚拟环境,查看进度,当然也可以在训练后看总结:
```
(tensorflow1) D:\tf\models\research\object_detection>tensorboard --logdir=images
```

### 7. 导出推断图
在object_detection目录中执行这个命令, 命令中的 "XXXX" 替换成 "model.ckpt-XXXX" 中最大数字的那个.:
```
python export_inference_graph.py --input_type image_tensor --pipeline_config_path training/faster_rcnn_inception_v2_pets.config --trained_checkpoint_prefix images/model.ckpt-XXXX --output_directory inference_graph
```
他在object_detection\inference_graph目录下创建了一个 frozen_inference_graph.pb 文件,这就是我们的目标文件.

### 8. 赶紧跑一下新模型.
具体参考例子源码.

![Sample](https://www.lijingquan.net/wp-content/uploads/2019/05/img_5cd17df0b5e73.png)

