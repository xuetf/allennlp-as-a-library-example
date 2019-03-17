#!/usr/bin/env bash

mode=$1
task_name=text_classifier

if [ $mode == 'train' ]
    then
        python3 -m allennlp.run \
        train experiments/${task_name}/venue_classifier_local.json \
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
        --predictor paper-classifier \
        --include-package libraries \
        --title "Academic Paper Classifier" \
        --field-name title \
        --field-name paperAbstract

fi
