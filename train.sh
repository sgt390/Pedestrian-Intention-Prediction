python3 -m intention_prediction.train --dataset ./dataset/ --timestep 15 --min_obs_len 8 --max_obs_len 12 --loader_num_workers 1 \
--num_epochs 50 --batch_size 32 --embedding_dim 128 --h_dim 32 --num_layers 1 --mlp_dim 64 --dropout 0.5 \
--batch_norm 0 --learning_rate 0.0005 --output_dir ./models --print_every 100 --checkpoint_every 30 --checkpoint_name \
cnnlstm_pie_standardcrop --checkpoint_start_from None --restore_from_checkpoint 1 \
--use_gpu 1 --gpu_num 1\

