#!/bin/bash

declare -A train_set_sizes=(["en_XX"]=62650 ["cs_CZ"]=42210 ["ru_RU"]=4980 ["de_DE"]=19237 ["ar_AR"]=19411 ["ro_RO"]=7082 ["zh_CN"]=1220676)

EP=2
BATCH_SIZE=4
MAX_LENGTH=128
for LANG in en_XX
do
	OUTPUT_DIR=mbart-$LANG
	for LR in 1e-5
	do
		for WARMUP in $(( ${train_set_sizes[$LANG]} / 8 ))
		do
			for SEED in 10
			do
				echo Lr: $LR warmup: $WARMUP seed: $SEED lang: $LANG		
				echo $LANG-$SEED-$LR-$WARMUP
				python ./transformers/examples/seq2seq/finetune.py --model_name_or_path=facebook/mbart-large-cc25 \
				--learning_rate=$LR \
				--train_batch_size=$BATCH_SIZE \
				--task gec \
				--data_dir=en \
				--src_lang en_XX \
				--tgt_lang en_XX \
				--langs=$LANG \
				--eval_batch_size=2 \
				--output_dir=$OUTPUT_DIR \
				--num_train_epochs=$EP \
				--warmup_steps $WARMUP \
				--freeze_embeds \
				--gpus=1 \
				--do_train \
				--val_metric loss \
				--val_check_interval 0.5 \
				--logger_name=default \
				--max_source_length $MAX_LENGTH --max_target_length $MAX_LENGTH --val_max_target_length $MAX_LENGTH --test_max_target_length $MAX_LENGTH \
				--seed=$SEED
				"$@"
								
				python evaluate.py $OUTPUT_DIR $SEED-$LR-$WARMUP $LANG 20 dev
				python evaluate.py $OUTPUT_DIR $SEED-$LR-$WARMUP $LANG 20 test				
				
				# rm -r $OUTPUT_DIR
			done
		done
	done
done