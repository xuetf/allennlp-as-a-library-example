#!/usr/bin/env bash

mode=$1
task_name=word2vec

if [ $mode == 'train' ]
    then
        python3 -m allennlp.run \
        train configs/${task_name}/sgns_text8.json \
        -s outputs/${task_name} \
        -f \
        --include-package libraries

elif [ $mode == 'test' ]
    then python3 -m allennlp.run evaluate \
         outputs/${task_name}/model.tar.gz \
         data/${task_name}/test.jsonl \
         --output-file outputs/${task_name}/test_metric.json \
         --include-package libraries

elif [ $mode == 'service' ]
    then
        python3 -m allennlp.service.server_simple \
        --archive-path outputs/${task_name}/model.tar.gz \
        --predictor synonyms_predictor \
        --include-package libraries \
        --title "Wor2vec" \
        --field-name word \

fi
