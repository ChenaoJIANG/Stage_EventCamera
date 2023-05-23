# Stage_EventCamera
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

