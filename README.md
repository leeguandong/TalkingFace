[📘使用文档]() |
[🛠安装教程]() |
[👀模型库]() |
[🆕更新日志]() |
[🚀进行中的项目]() |
[🤔报告问题]()

## 电商场景多模态大语言模型
本系列主打代码理解和快速需求完成，然后快速集成简单应用，在这个系列中集成多个多模态大语言模型，可能会涉及推理等一系列工程，训练是核心模块之一，这一点要不同于sd系列     

### 设计理念    
单模型文件策略+非侵入式
1.single model file policy,单模型文件策略，任何模型的所有代码都只应该放在一个文件中，这个文件就是该模型自己的模型文件,拒接任何将不同模型的相同子模块抽象并集中到一个新文件中的尝试，不想要一个包含所有可能得注意力机制的attention_layer.py，我认同这种关系，mm系列的代码到最后的维护成本太高了。
Huggingface: https://mp.weixin.qq.com/s/cjsukNonWn9tIsud4Y07Zg 
2.tmux https://zhuanlan.zhihu.com/p/98384704   
tmux new -s xxx
tmux detach
tmux attach -t xxx
     


### 模块内容
- **configs**   
   配置模板


### 接口的gpu显存和利用率   
平台：autodl T4 cuda 11.8 dirver 525.105.17 权重：v1-5-pruned-emaonly.safetensors 默认：autocast    


### 技术实现难点
- **怎么实现模型更换？**    
    包括lora模型，实现模型的可选参数，如何设置？
- tensorrt不支持int64的权重，因此可先将int64的onnx转成int32位的onnx，可以用onnxruntime默认的tensorrtprovides进行推理



### install
1.git clone -b v0.8.3 https://github.com/microsoft/DeepSpeed.git
cd DeepSpeed
pip install -e.




### 参考资料


### 训练
1.diffusers训练的权重很多放在/root/.cache/huggingface/hub





