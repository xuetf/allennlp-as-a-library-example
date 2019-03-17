# Input

1. Vocabulary初始化: 既从instances读取，又可以从params中指定的文件中读取。两者都指定时，可以extend在一起。
   
2. Vocabulary的命名空间：从instances的所有的fields中读取，主要是count_vocab_items方法，不同fields实现不同。
   比如，TextField是从tokenIndexers Dict成员的keys中得来的；而LabelField是直接从label_namespace成员来的，
   SequenceLabelField也是从label_namespace成员来的。
   
3. 特别地，对于TextField，可以定义tokenIndexer，tokenIndexer有很多种默认实现，比如bert-pretrained，在
   woedpiece_indexer.py中的PretrainedBertIndexer，该方法中自带了原始Bert模型中使用的vocabulary，重载
   的count_vocab_items为空，因此textField本身的词token以及namespace不会抽取出来放在train标准过程的vocabulary中。
   但是仍然可以索引index，即PretrainedBertIndexer重写了tokens_to_indices，并且用的是自身的Bert的vocabulary进行index的。
   即使tokens_to_indices传入了vocabulary形参(标准train过程从Instance中得到的)，也直接忽略，而是使用自身的Bert的vocabulary。
   
   