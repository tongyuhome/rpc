# RPC

最近看到旷视南京研究院发布的一个新零售自动结算场景下的大型商品数据集RPC（[Project](<https://rpc-dataset.github.io/>)，[Paper](<https://arxiv.org/abs/1901.07249>)）。

其中利用GAN对人造数据进行渲染前后所训练出的检测器的检测效果有较大提升，因此想尝试复现一下论文中所展示的效果。

先对论文中的内容做一下简单的梳理：

## 数据集

这个数据集一共拍摄了200种商品，在项目中有提供数据集，内容包括：

train2019,val2019,test2019三个文件夹，分别包含了53739(8.54GB)，6000(1.23GB)，24000(4.95GB)个文件，均为JPG格式图像。训练数据集中每个图像包含了一件商品，置放在一个圆形展示台上，根据不同的摄像头位置和不同的圆台转动角度来分别拍摄。![拍摄商品](<https://github.com/tongyuhome/rpc/raw/master/show_images/take_pic.png>)

验证和测试数据集中每个图像包含了多个商品，置放在纯白色台面，根据摆放商品的个数和种类被分为了三个等级：Easy ，Medium ，Hard 。

| Clutter levels | categories | instances |
| -------------- | ---------- | --------- |
| Easy mode      | 3-5        | 3-10      |
| Medium mode    | 5-8        | 10-15     |
| Hard mode      | 8-10       | 15-20     |

验证数据集：

![val_easy](<https://github.com/tongyuhome/rpc/raw/master/show_images/val_1999.jpg>) ![val_medium](<https://github.com/tongyuhome/rpc/raw/master/show_images/val_4000.jpg>) ![val_hard](<https://github.com/tongyuhome/rpc/raw/master/show_images/val_6000.jpg>)

测试数据集：

![test_easy](<https://github.com/tongyuhome/rpc/raw/master/show_images/test_8000.jpg>) ![test_medium](<https://github.com/tongyuhome/rpc/raw/master/show_images/test_16000.jpg>) ![test_hard](<https://github.com/tongyuhome/rpc/raw/master/show_images/test_24000.jpg>)

和instances_train2019,instances_val2019,instances_test2019三个json格式文件，每个json文件都包含了info，licenses，categories，__raw_Chinese_name_df，images，annotations六种信息。

对于前四种信息，三个json文件的内容都是一样的，info和licenses记录了这个项目的一些基本信息，categories和raw_Chinese_name_df则记录了200种商品的基本信息：分别展示[‘categories’]-[0]和[‘raw_Chinese_name_df’]-[0]
{'supercategory': 'puffed_food', 'id': 1, 'name': '1_puffed_food'}
{'sku_name': '1_puffed_food', 'category_id': 1, 'sku_class': 'puffed_food', 'code': 6909409012031, 'shelf': 1, 'num': 4, 'name': '上好佳荷兰豆55g', 'clas': '膨化食品', 'known': True, 'ind': 0}

images和annotations则是对每张图的内容做一个说明和注释：分别展示[‘images’]-[0]和[‘annotations’]-[0]

{'file_name': 'xx.jpg', 'width': xx, 'height': xx, 'id': xx}
{'area': xx, 'bbox': [xx, xx, xx, xx], 'category_id': xx, 'id': xx, 'image_id': xx, 'iscrowd': xx, 'segmentation': [[]], 'point_xy': [xx, xx]}

## 实验

在说实验之前，先提及一下论文中提出的几个指标，这是判断检测器效果好坏的客观依据。

先定义各符号所表示的意思。从$K$类商品中选出N件商品，$P_{i,k}$ 表示第$i$张图中$k$类商品的预测数量，$GT_{i,k}$表示第$i$张图中$k$类商品的真实数量，$CD_{i,k}$表示$P_{i,k}$和$GT_{i,k}$之间的$l_{1}$距离，能反映出图中某一类别商品的错误计数，$CD_{i}$反映第$i$张图中所有$K$类商品的预测误差，$CD_{i}$为0的话表示预测完全正确。
$$
CD_{i,k} = |P_{i,k} - GT_{i,k}| \\
CD_{i} = \sum_{k=1}^{K}CD_{i,k}
$$
**Checkout Accuracy (cAcc):**结账准确率，表示检测过程中$CD_{i} = 0$发生的概率。$\delta()$当且仅当$\sum_{k=1}^{K}CD_{i,k}=0$时为$1$否则为$0$，所以$cAcc$的取值范围是$[0,1]$
$$
cAcc=\frac{\sum_{i=1}^{N}\delta(\sum_{k=1}^{K}CD_{i,k},\quad0)}{N}
=\frac{\sum_{i=1}^{N}\delta(CD_{i},\quad0)}{N}
$$
**Average Counting Distance (ACD):**平均计数距离，表示每个图像的平均计数错误。
$$
ACD=\frac{1}{N}\sum_{i=1}^{N}\sum_{k=1}^{K}CD_{i,k}=\frac{1}{N}\sum_{i=1}^{N}CD_{i}
$$
**Mean Category Counting Distance (mCCD):**平均类别计数距离，表示每个商品类别的计数误差的平均比率。
$$
mCCD=\frac{1}{K}\sum_{i=1}^{N}\frac{\sum_{i=1}^{N}CD_{i,k}}{\sum_{k=1}^{K}GT_{i,k}}
$$
**Mean Category Intersection of Union (mCIoU):**平均类别交并比，表示每个类别预测值和真实值之间误差的平均值，就是每个类别检测结果IoU的平均值。
$$
mCIoU=\frac{1}{K}\sum_{i=1}^{N}\frac{\sum_{i=1}^{N}min(GT_{i,k},P_{i,k})}{\sum_{i=1}^{N}max(GT_{i,k},P_{i,k})}
$$
除了以上四个指标，论文中还引用了$mAP50$和$mmAP$两个指标来客观验证检测效果的好坏。

论文中设置了四种不同的基线实验Single，Syn，Render和Syn+Render，这四个实验实际上是使用四种不同的数据集去训练相同的检测器。

检测器：用特征金字塔网络 ([FPN](https://arxiv.org/abs/1612.03144)) 作为检测器.

Single：使用单个商品的数据集去训练。直接用单个商品的图像去训练检测器。

Syn：使用合成图像去训练。将单个商品图像中的商品抠出来，按照一定要求复制黏贴到空白背景中，合成模拟图像。

Render：使用渲染图像去训练。对合成好的图像进行渲染，补充其丢失的光影特征。

Syn+Render：用合成图像以及渲染图像一起去训练。

下图是论文中展示的流程图，我正是根据这个流程图来复现论文中的效果，其过程在中有详细记录。

![Pipeline](<https://github.com/tongyuhome/rpc/raw/master/show_images/Pipeline.png>)

