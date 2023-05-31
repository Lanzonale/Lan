# Lan
Cifar-100 classifier

main为baseline主程序，从三个data argumentation python程序里导入cutmix，cutout，mixup可以进行相对的预处理，调整各项超参数后即刻进行训练，最终模型此处预设以h5格式保存。
testing为测试用程序，更改读入模型可更改测试对象，结果返回预测的标签类别及真实类别。

