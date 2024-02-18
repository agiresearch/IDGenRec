python ../src/main_generative.py --datasets Beauty \
  --distributed 1 \
  --gpu 0,1,2,3 \
  --tasks sequential \
  --item_indexing generative \
  --rounds 3\
  --id_batch_size 12 \
  --rec_batch_size 64 \
  --master_port 1993 \
  --prompt_file ../prompt.txt \
  --sample_prompt 1 \
  --eval_batch_size 1 \
  --dist_sampler 0 \
  --max_his 20  \
  --sample_num 3 \
  --train 1 \
  --test_prompt seen:0 \
  --rec_lr 1e-3 \
  --id_lr 1e-8 \
  --test_epoch_id 1 \
  --test_epoch_rec 10 \
  --his_prefix 0 \
  --random_initialize 0 \
  --id_epochs 1 \
  --rec_epochs 10 \
  --alt_style id_first
  # --rec_model_path ...
  # --id_model_path ...