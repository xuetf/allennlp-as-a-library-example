不同任务的执行脚本，每种任务分为了:

## 3种模式
- train训练
    
   对训练集进行训练，训练过程中可以使用tensorboard实时查看训练进展。
   tensorboard --logdir outputs/${task_name}/log/


- test测试
    
   对测试集进行测试。


- service服务
    
   会在localhost起一个网址，在该网址中可以输入，并使用模型实时输出。
   
每种模式具体的命令，可以直接修改.sh文件。
   
## 示例
   
实际运行时，只需要传入上述3种参数。例如，文本分类任务：

训练：script/text_classifier.sh train

测试：script/text_classifier.sh test

服务：script/text_classifier.sh service
