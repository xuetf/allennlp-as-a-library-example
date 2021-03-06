#!/usr/bin/env bash

mode=$1
task_name=pos_tagger

if [ $mode == 'train' ]
    then
        python3 -m allennlp.run \
        train configs/${task_name}/experiment_conll2000.json \
        -s outputs/${task_name} \
        --include-package libraries \
        -f


elif [ $mode == 'test' ]
    then python3 -m allennlp.run evaluate \
         outputs/${task_name}/model.tar.gz \
         data/${task_name}/test.txt \
         --output-file outputs/${task_name}/test_metric.json \
         --include-package libraries


elif [ $mode == 'service' ]
    then
        python3 -m allennlp.service.server_simple \
        --archive-path outputs/${task_name}/model.tar.gz \
        --predictor sentence-pos-tagger \
        --include-package libraries \
        --title "Part-of-Speech Tagger" \
        --field-name sentence

fi
