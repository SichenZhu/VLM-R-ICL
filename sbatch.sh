#!/bin/bash

# model="LLaVA-CoT"
# model="Llama-3.2-11B-Vision-Instruct"
# model="VLM-R1"
# dataset="MathVista_MINI"
# dataset="EMMA"
# dataset="ScienceQA_TEST"
# query_dataset="ScienceQA_TEST_QCME"  # _wo_last
# dataset="A-OKVQA"
# dataset="MME_CoT_TEST"
# dataset="LogicVista"
# dataset="LogicVista_Rationale"  # _wo_last
# dataset="TextVQA_VAL"

# support_dataset="TextVQA_TRAIN"
# query_dataset="TextVQA_VAL"

# support_dataset="ScienceQA_TRAIN"
# support_dataset="ScienceQA_TRAIN_correct"
# support_dataset="ScienceQA_TRAIN_QCME"
# support_dataset="ScienceQA_TRAIN_QCME_correct"
# query_dataset="ScienceQA_TEST"

# support_dataset="A-OKVQA_TRAIN"
# support_dataset="A-OKVQA_TRAIN_correct"
# support_dataset="A-OKVQA_TRAIN_QCME"
# support_dataset="A-OKVQA_TRAIN_QCME_correct"
# query_dataset="A-OKVQA_VAL"

# support_dataset="A-OKVQA_TRAIN"
# query_dataset="A-OKVQA_VAL"
# support_dataset="M3CoT_TRAIN"
# query_dataset="M3CoT_TEST"

# rag_method="jices"


cd /data1/szhu337/VLM_proj/VLM-R-ICL

export model="VL-Rethinker-72B"
export icl_rationale=1 # 0: icl_rationale=False (base model), 1: icl_rationale=True (reasoning model)
# export model="Qwen2.5-VL-72B-Instruct"
# export icl_rationale=0

for support_dataset in "M3CoT_TRAIN"
do
    for query_dataset in "M3CoT_TEST"
    do
        for rag_method in "random" # "random" "jices"
        do
            for shot in 0 1 2 4 8
            do
                export support_dataset=$support_dataset
                export query_dataset=$query_dataset
                export rag_method=$rag_method
                export shot=$shot
                bash scripts/srun.sh
            done
        done
    done
done


# 0 1 2 4 8