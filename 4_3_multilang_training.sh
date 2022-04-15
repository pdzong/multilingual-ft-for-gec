#!/bin/bash

declare -A train_set_sizes=(["en_XX"]=62650 ["cs_CZ"]=42210 ["ru_RU"]=4980 ["de_DE"]=19237 ["ar_AR"]=19411 ["ro_RO"]=7082 ["zh_CN"]=1220676)
TRAIN_SET_SIZE=0
for LANG in en_XX cs_CZ de_DE ru_RU ro_RO ar_AR zh_CN
do
	TRAIN_SET_SIZE=$(( $TRAIN_SET_SIZE + train_set_sizes[$LANG] ))
done

EP=2
BATCH_SIZE=$3
MAX_LENGTH=128
OUTPUT_DIR=$2
LANGS=en_XX,cs_CZ,de_DE,ro_RO,ru_RU,ar_AR,zh_CN
for LR in $4
do 
	for WARMUP in $(( $TRAIN_SET_SIZE / ( $EP * $BATCH_SIZE ) ))
	do
		for SEED in 80
		do
			echo Lr: $LR warmup: $WARMUP seed: $SEED			
			python ./transformers/examples/seq2seq/finetune.py --model_name_or_path=$1/$2 \
			--learning_rate=$LR \
			--train_batch_size=$BATCH_SIZE \
			--task gec \
			--data_dir=en \
			--src_lang en_XX \
			--tgt_lang en_XX \
			--langs=$LANGS \
			--eval_batch_size=$BATCH_SIZE \
			--output_dir=$OUTPUT_DIR \
			--num_train_epochs=$EP \
			--warmup_steps $WARMUP \
			--freeze_embeds \
			--gpus=1 \
			--do_train \
			--val_metric loss \
			--val_check_interval 1.0 \
			--logger_name=default \
			--max_source_length $MAX_LENGTH --max_target_length $MAX_LENGTH --val_max_target_length $MAX_LENGTH --test_max_target_length $MAX_LENGTH \
			--seed=$SEED
			"$@"

			python evaluate.py $OUTPUT_DIR $SEED-$LR-$WARMUP-$1-$2 $LANGS 20 dev
			python evaluate.py $OUTPUT_DIR $SEED-$LR-$WARMUP-$1-$2 $LANGS 20 test
			# rm -r $OUTPUT_DIR

		done
	done
done