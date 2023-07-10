# Stage_EventCamera  日期为汇报日， 记录汇报之后的任务
The research objective of the internship is to segment moving objects from event-based vision in the context of driving scenes

Le processus de stage:
第一个月，看event camera，motion segmentation相关论文，归纳整理不同的方向和方法（model，ML...）
找相关code并测试，试用DAVIS event camera，看博士的论文和测试他的code.

3.29开始，尝试EVIMO论文方法，code在github上DVS
配置环境 conda evimo， torch
修改输入，input为三通道2dmap（后来在betterflow/evimo中找到了转换成该input格式的code）4.7 生成map速度太慢，需要优化，优化后不错，但自己写的代码很难匹配

4.11 发现DVS里有很多分支，有一些有用的code
开始训练网络，但需要训练20h

4.19 尝试使用cassio02来训练，学习tmux和sftp
4.26 得到了跟文章类似的mask预测（但文章中的预测都是训练集的，训练集训练，训练集test，自欺欺人！）但depth预测不好，原因是ground truth的depth只有objet的部分，不包括背景，所以并不能很好的训练
尝试联系betterflow/evimo的作者问depth，得到depth确实不包括背景的回答，可以通过rosbag修改参数regenerer带有背景的depth，但在我的电脑上运行out of memory，cassio02上不能安装ros（尝试docker，失败）。
作者也说depth不太准确，需要考虑是否有意义花时间在这一步上。

5.3 放弃重新生成EVIMO1带背景的depth，没有太大意义，尝试新版本数据集EVIMO2v2，但EVIMO2有很多objets，其中包括一些没有动的objet，所以直接训练会将所有物体的mask都作为gt训练，跟我的实习主题并不匹配
所以尝试能否将未动的objet的mask去掉，只留下移动物体的mask。meta.txt文件中有每一帧每一个物体以及camera的pose，将每个物体的pose和camera的pose进行转化，变成世界坐标系下的pose，
再选出小于一定速度的objet，即未动的物体，将他们的mask标记为0（即去除），每一个objet的mask都是该物体编号乘以1000（比如，1号objet的mask为1000）

5.10 用新数据集训练网络，但很玄学，参数一样，有的训练的好，有的不好；两个数据集图片的分辨率不一样，不能一起作为input训练，evimo1预训练在evimo2训练效果也不好，目前只用evimo2训练

5.17 只有一个只用evimo2训练的网络效果不错，其余的（光流有问题，depth有问题）不好，但目前的网络只能用于分割移动物体，但如果有多个物体也只能将其与背景分割开（同一个mask），并不能将不同物体分开（比如用不同颜色的mask分割不同objet）
研究multi-task的论文，尝试优化网络，比如使用现有网络得到的光流，depth以及gt mask作为input，用multi task网络增加分割不同物体的新task

5.24 网络结构了解不清，开始细致的研究网络细节，画网络图，研究loss

5.31 pixelpose的网络改进，输出residuel的信息 是xyz的速度信息，希望通过聚类算法进行聚类；尝试使用kmeans

6.7 研究seg方面以及聚类的论文；合并速度信息和坐标信息使用kmeans有进步，但需要定义k且无监督，放弃；光流的color wheel显示
了解几大类分割，以及其不同的metric

6.14 两个option，1.用pose网络，使用mask进行语义分割（如segnet）； 2.用pixelpose网络，使用速度和位置信息进行聚类，需要自己写网络，输入是信息流（不用是image）
先尝试第一个option；由于数据集中很多物体都没动也没出现，有一些类没有样例，会影响别的类；并且不能只识别运动物体，所以尝试用multi task learning，一个task是motion mask，另一个语义分割
但还是数据集的问题，以及multi task loss权重的设计 需要进一步改进；
将数据集没出现的类删掉（29变为12类）
设计loss自适应权重

6.21 继续删除不需要的类（12变7类）确认两个同方向移动的物体可以语义分割开（原来用聚类不太行）
该交叉熵损失函数里，每个类的权重，背景占比很大，权重就得小
数据增强：随机反转，随机缩放，随机剪切...
  语义分割的gt图像包括labels，放大缩小需要使用Nearest-neighbor interpolation，而不能用初始的插值
先手动调整每个损失函数的权重（depth，bg，label：1 1 1 or 0.5 5 0.4）结果差不了太多，但111更稳定，泛化也好一点（gt缺失的时候，预测也应该能预测出来）
尝试自适应权重的方法；Dynamic Weight Average(dwa)：保证每个任务的速度相近，loss变化大，weight就变小
GradNorm要算gradient，计算量大，保证每个任务的速度和量级都相近，111效果已经不错了，太复杂没必要

6.28 了解tensorboard怎么用，add scalar， add image， add graph等等，远程gpu上怎么用本地浏览器看实时tensorboard网页
训练集，验证集，测试集重新分配；
目前网络共用encode和decode，因为目前这种两个预测会互相影响，语义分割也会只在物体动的时候分割
（共享en/de，对于测试集效果不好，因为semantic不好同时也影响了motion seg）
（共享en，motion seg对测试集效果不错，但semantic还是不行）
尝试只共用encode，写一个简单的decode来语义分割；比较一下那种更好；可以将语义的gt用原始的gt（no moving）
选用mIoU来评估（对于semantic）

7.5 将semantic的gt改用原始的gt（no moving），看看网络会不会有什么interesant
要将dataloader.py大改！ 
改良之后测试集只能在部分scene效果较好 miou=0.4/0.7 这部分scene是样本较多的
而且目前没有人在evimo2上测试，没有比较
所以又重新在evimo1上用multi-task网络测试


















