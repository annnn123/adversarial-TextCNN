# adversarial-TextCNN
文本分类 + 对抗训练，实现《FAST IS BETTER THAN FREE:REVISITING ADVERSARIAL TRAINING》中的PGD、Free、FGSM。

## 环境
    python 3.6.5
    pytorch 1.8.1
    
## 数据集
    以MR数据集为例（train/test：7108/3554): 
    原始数据：./dataset/MR/MR.txt
    标签：./dataset/MR/MR_label.txt
    清洗后的数据：./dataset/MR/MR_clean.txt
    训练集、测试集：./dataset/MR/train.json, ./dataset/MR/test.json
    词表：./dataset/MR/vocab.json

## 使用说明
    通过修改config.py中的配置参数adv_method实现不同的对抗训练方法：
    adv_method='': 不使用对抗训练
    adv_method='PGD': TextCNN+PGD
    adv_method='Free': TextCNN+Free
    adv_method='FGSM': TextCNN+FGSM
    
    训练（earlystop） + 测试：
    python train.py
    
    其他的TextCNN参数、训练参数、数据集参数、以及对抗训练超参数（eps、alpha、PGD_steps、Free_num_replays）均在config.py中修改。
    
## 参考
    《EXPLAINING AND HARNESSING ADVERSARIAL EXAMPLES》
    《ADVERSARIAL TRAINING METHODS FOR SEMI-SUPERVISED TEXT CLASSIFICATION》
    《FAST IS BETTER THAN FREE: REVISITING ADVERSARIAL TRAINING》
     https://github.com/649453932/Chinese-Text-Classification-Pytorch
     https://github.com/locuslab/fast_adversarial

    
    
    
