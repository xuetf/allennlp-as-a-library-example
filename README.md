# Different NLP task using AllenNLP to implement
    
整理了使用AllenNLP实现不同NLP任务的代码。大部分来自于AllenNLP的默认实现，把代码抽取出来并重新组织。

## NLP Tasks

- **文本分类任务**（Text Classifier)

    - 1. [参考](https://github.com/allenai/allennlp-as-a-library-example) 
    
    - 2. 将Paper划分到不同领域（ACL,ML）。


- **词性标注/句法结构分析** (Pos Tagger/Phrase Structure Parsing)

    - 1. 对语句进行词性标注或进行短语结构分析。
    - 2. 参考AllenNLP的默认实现。
   
    
- 未完待续...


## Project Structure

- data：各任务训练、验证、测试数据等。

- experiments：各任务训练的.json配置文件，训练的核心入口文件。

- librarys: 各任务代码，主要包括数据读取/模型/预测器(对外提供服务时使用)

- tests: 各任务的单元测试代码。
    
- outputs：各任务训练时的输出，例如checkpoint，vocabulary等。在线服务,tensorboard查看等都需要基于该文件夹。
   
- scripts: 各任务实际的训练/测试/服务shell脚本，直接运行即可。