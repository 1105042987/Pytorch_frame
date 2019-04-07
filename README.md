# 使用说明

### train.py

- train.py -h 获取帮助
- 新增网络
  - 字典 net_dic 与函数 refresh_net_dic(net,arg) 不可更改其名称；
  - 新添加网络时，在 net_dic 中如已有网络所示注册，注册后将在参数 -n 中允许使用；
  - 若需要通过命令行调整网络参数：
    - 在函数 get_args() 中注册一行参数获取；
    - 在函数 refresh_net_dic(net,arg) 中使用 arg\['name'\]\[x\] 更新参数；

- 加载数据集
  - 在 dataset 文件夹下创建一个文件夹存储数据集；
  - 在该文件夹中创建 dataset.py 文件：
    - 继承 torch.utils.data.Dataset 进行数据加载与预处理；
    - 实现 loadData 函数进行每个 epoch 的数据增强；

- 错误率计算
  - 继承 FR.weak_ErrorRate 虚类
    - 实现 add(self,out,tar) 函数，更新 self.error, self.total两个参数；
- 从无监督数据中获取预测答案
  - 继承 FR.weak_GetPredictAnswer 虚类
    - 实现 add(self,outputs,index) 函数，把每个batch的结果以一定格式存储；
    - 实现 save(self,name) 函数，以与 name 相关的名字，以一定数据类型输出；



### evaluate.py

- evaluate.py -h 获取帮助
- get_args() 函数中增加额外需要的选项