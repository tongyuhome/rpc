# 复现实验
记录实验过程：思路，实现方法，问题，解决/为解决。

四个基线实验中，第一个实验直接使用单个商品图像数据来训练，这个在数据集中直接有，而第四个实验是基于2，3实验来完成的，所以要处理的是实验2，3中所需数据的获取。

首先第一步要从单个商品图像中将商品目标分割出来，大致可分为以下几步：

1. 利用所给信息对单个商品图像进行bounding box的标注并以此进行剪裁（当然实际操作可以直接剪裁），分别剪裁出一倍和两倍bounding box大小的图像。
2. 对两倍bounding box大小的图像用显著性检测，将前后景区分开来。
3. 将2.中的结果与一倍bounding box大小的图像一起通过CRF进行细化处理，再将其按照一倍bounding box大小进行剪裁，得到最后的mask。
4. 将3.中得到的mask再与一倍bounding box大小的图像一起结合抠出目标图像，最后将黑色背景透明化处理即可。

下图是一个实现流程草图（展示图是我得出的效果）：

![分割目标](show_images/Snipaste_2019-05-16_11-44-52.png)具体的实现过程：

1. 这一步的剪裁没有什么问题，需要注意的是因为拍摄位置关系有的商品图像的两倍bounding box区域会超出原图，这时候以边界为准进行剪裁即可，同时记录一下剪裁位置，方便对之后CRF得到的MASK进行剪裁。当然，二倍bounding box剪裁的作用在于为显著性检测提供前后景内容，所以如果进行二倍bounding box剪裁的时候超出原图范围也可以少剪裁一些，这样之后对MASK进行剪裁的时候只需要有bounding box大小的数据即可。
2. 关于显著性检测的方法，论文中提及了使用 [Detecting salient
   objects via color and texture compactness hypotheses](<https://ieeexplore.ieee.org/abstract/document/7523421>)中的方法来获取初步的MASK，由于这篇论文未公布代码，所以我在GitHub上找了一份[显著性检测](<https://github.com/Joker316701882/Salient-Object-Detection>)的代码来实现，关于显著性检测的研究有很多，所以这部分有一些前人的成果可以借鉴。我的得到的效果由于目标颜色与背景颜色相似程度以及光亮相似程度使得分割出的MASK会或多或少一些，这样的状况在论文中也有体现，初步MASK的生成会直接影响最后分割出商品效果的完整度，对之后的检测有多少影响这个就需要实验来验证了，主观的猜测应该还是蛮有影响的，因为之后如果要实际应用的话抠出来的商品MASK越接近真实商品肯定是越好的。我现在使用的是预训练好的模型来进行显著性检测，如果能够用需要检测的的同类型进行训练，那模型大概率能得到更好的MASK图，但是这样就需要真实MASK图来进行训练，这一步的成本消耗较高，比较麻烦。
3. 关于CRF的方法，论文中提及了使用[Efficient inference in fully
   connected CRFs with Gaussian edge potentials](<http://papers.nips.cc/paper/4296-efficient-inference-in-fully-connected-crfs-with-gaussian-edge-potentials.pdf>)中的方法来对2.中生成的初步MASK图进行细化处理，同样由于论文没有提供源码，我也在GitHub上找了一份[CRF](<https://github.com/Andrew-Qibin/dss_crf>)的代码来实现，可以将初步MASK中边缘模糊的内容变得清晰化，这个对MASK图的边缘细化效果很好，但是如果MASK图本身是有缺失的，那CRF这样的处理会使得缺失增大，当然处理这个问题最直接的还是改善显著性检测得到MASK图的质量。
4. 这一步是将同为一倍bounding box大小的的商品图和MASK图进行一个累加，对MASK部分进行填充，之后再将背景中的黑色部分透明化处理即可。

下图是论文中实现的一个例子(左侧方框)，以相同的原图我所得到的效果(方框右侧)：

![segmenting](show_images/seg_res.png)

a1

