# 旷视AI智慧交通开源赛道-交通标志识别-仙交小分队方案

* 队伍：仙交小分队 

* 决赛得分：0.54674

本次的比赛是交通标志检测，交通标志本身种类众多，大小不定，并且在交通复杂的十字路口场景下，由于光照、天气等因素的影响，使其被精确检测变得更加困难。通过反复实验，我们选择了今年旷视新出的YOLOX目标检测框架作为baseline，这里是官方代码 [MegEngine YOLOX implementation](https://github.com/MegEngine/YOLOX)。megengine版本的yolox代码相对于pyotorch版本的bug较多，通过反复调试，我们最终的方案是：YOLOX_L + P6 + Focalloss + Inputsize2048 + 双线性插值上采样 + dataaug30，以下是整个调试的过程。

yolox-l在coco上的预训练megengine模型以及我们本次比赛的权重文件可以从下列百度网盘的链接中获取，其中best_ckpt.pkl请放在`YOLOX_outputs/yolox_l`目录下，yolox_l_mge.pkl（或者通过pytorch版本的yolox自行转化coco的预训练权重）放在`pretrained`目录下。

> [百度网盘链接](https://pan.baidu.com/s/12HZ6nSBvaXiFjedQ5ryktQ )         提取码：l8yh

## 数据分析
本次比赛提供的数据集，图片总数为3349张，长的均值为1680.0883845924157， 宽的均值为2227.9558077037923，训练集和验证集一共有2700张，初赛的测试集有580张。

详细的规模如下：

```
(h, w, c): num
(1200, 1600, 3): 941

(1944, 2592, 3): 1259

(2048, 2448, 3): 18

(1080, 1920, 3): 32

(1520, 2704, 3): 511

(2048, 2048, 3): 582
```

数据一共包含5种类型的目标，分别是：红灯、直行标志、向左转弯标志、禁止驶入和禁止临时停车，目标的数量如下

```
0:red_tl--1465个
1:arr_s--1133个
2:arr_l--638个
3:no_driving_mark_allsort--622个
4:no_parking_mark--1142个
```

目前的数据集来看，第三类的目标比较少，还是和之前一样，毕竟是数据驱动的，我们最好还是能通过copy-paste这种方法在数据上做一些提升。

具体数据的例子如下，从下面的图中可以看出，目标基本都比较小，而且还存在很多遮挡的情况。

![](https://vehicle4cm.oss-cn-beijing.aliyuncs.com/typoraimgs/3279125,1bb5780005b1590b9.jpg)

![](https://vehicle4cm.oss-cn-beijing.aliyuncs.com/typoraimgs/image-20210916135049212.png)

## Yolox的配置

比赛中要求使用的框架是megengine，其中yolox有两个实现版本，一个是[pytorch](https://github.com/Megvii-BaseDetection/YOLOX)版本的，一个是[megengine](https://github.com/MegEngine/YOLOX)版本的，其中pyotrch版本的代码相对于megengin版本的代码要相对更加完善一些，并且在pytorch中提供了大量的coco预训练模型，这些预训练模型也可以转化为megengine的版本做finetune使用。

### megengine-yolox的安装

* 安装并激活虚拟环境

  首先是虚拟环境的设置，我们这里创建的是python版本为3.7.10，名字为xuan的虚拟环境（主要是为了和megstudio中的保持一致），命令如下：

  ```
  conda create -n xuan python==3.7.10
  conda activate xuan
  ```

* 安装megengine

  最重要的是megengine的安装（！！！这里一定要安装1.4.0的版本，1.5.0的话训练的时候会有bug），如果是在megstudio平台上使用的话，可以方便的选择megengine的版本，直接选择1.4.0的版本即可，如果是在本机上安装使用的话，需要确保在虚拟环境中的cuda是10.1的版本，因为目前megengine提供的安装包是10.1的版本，命令如下：

  ```
  conda install cudatoolkit=10.1
  conda install cudnn==7.6.4
  pip install megengine==1.4.0 -f https://megengine.org.cn/whl/mge.html
  ```

* 安装yolox

  下面就是yolox的安装了。首先需要从官网把代码下载下来，实在觉得慢的话就直接下载安装包然后解压就完事了。

  ```
  https://github.com/MegEngine/YOLOX.git
  ```

  安装的命令如下

  ```
  cd YOLOX
  pip install -r requirements.txt # megengine记得注释掉，因为已经安装完了
  pip install torch torchvision # 虽然在实际训练过程中没有使用到，但是为了能安装通过还是需要安装一下这个
  python setup.py develop
  pip install cython
  pip install pycocotools
  ```

  安装完了之后记得打开python解释器，import yolox试试，没有报错的话基本就没啥问题了。

* 执行demo代码

  从官方下载一下yolox的tiny模型，放在当前目录的`pretrained`目录之后执行下面的命令，如果能够在yolox_outs下输出一张这样的检测结果，说明代码就没啥问题了

  ```
  python tools/demo.py image -n yolox-tiny -c /pretrained/yolox_tiny.pkl --path assets/dog.jpg --conf 0.25 --nms 0.45 --tsize 416 --save_result --device [cpu/gpu]
  ```




## 使用yolox对交通标志数据集进行微调

微调的过程还是比较痛苦的，因为自己的基础实在是太差，好多地方自己不是很熟悉，导致中间debug花费了很多时间，就很离谱

### 模型转移和验证

由于官方的megengine版本的yolox代码只提供了tiny版本的预训练模型，tiny版本的预训练模型速度虽然比较快，但是精度相对较低，想要在比赛中拿到比较好的成绩的话，还是得用l或者x版本得预训练模型，这样就只能从官方的pytorch代码中进行迁移。

* 模型迁移

  迁移的过程就不赘述了，大家看这里的官方文档就可以了，写的非常详细，现在yolox-l在coco上的预训练模型完成迁移即可。

  [YOLOX/demo/MegEngine/python at main · Megvii-BaseDetection/YOLOX (github.com)](https://github.com/Megvii-BaseDetection/YOLOX/tree/main/demo/MegEngine/python)

* 模型验证

  迁移完毕之后不要直接拿过来使用，首先要验证一下在coco数据集上的精度是否正确。

  首先需要建立软链接（软链接确实是个好东西，你可以不用来回地复制数据，非常节省空间），命令如下：

  ```bash
  ln -s /path/to/your/COCO ./datasets/COCO
  ln -s /root/data/traffic5 ./datasets/traffic5
  ```

  验证的命令如下，这里是验证官方代码的，其中fp16和fuse这个参数在megengine中是不能使用的

  ```bash
  python tools/eval.py -n  yolox-s -c yolox_s.pth -b 64 -d 8 --conf 0.001 [--fp16] [--fuse]
  # 验证tiny模型
  python tools/eval.py -n  yolox-tiny -c yolox_s.pth -b 4 -d 1 --conf 0.01           
  # 验证转移之后的yolox_l的megengine模型
  python tools/eval.py -n  yolox-l -c yolox_l_meg.pkl -b 4 -d 1 --conf 0.01 # 一般转移之后megengine的模型是pkl格式的
  ```
  
  如果验证完毕之后tiny在0.32左右，l在0.49左右，那就没啥问题了，要是在0.2或者直接是0.0，那说明你的模型转移就有问题
  
  ### 训练和提交

  整个模型采用了`YOLOX_L + P6 + Focalloss + Inputsize2048 + 双线性插值上采样 + dataaug30`的方案，具体的执行流程请查看`work.ipynb`，按照jupyter文件中的内容完成整个训练和测试的流程。

## 致谢

感谢比赛工作人员的辛勤付出和耐心答疑，感谢旷视提供了这次宝贵的参赛机会，从这次比赛中学到了很多关于目标检测的新东西，本次的方案是基于旷视开源的YOLOX，在YOLOX的官方群中看到了作者葛政认真回复大家的问题，也从作者刘松涛对论文的解读中体会到这个作品的来之不易，感谢之外更多的是敬佩，YOLOX赛高！

