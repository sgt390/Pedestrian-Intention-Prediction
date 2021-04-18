python3 -m intention_prediction.train-scenes --dataset ./dataset/ --timestep 15 --obs_len 8 --loader_num_workers 1 \
--num_epochs 50 --batch_size 32 --embedding_dim 128 --h_dim 32 --num_layers 1 --mlp_dim 64 --dropout 0.5 \
--batch_norm 0 --learning_rate 0.0005 --output_dir ./models --print_every 100 --checkpoint_every 30 --checkpoint_name \
cnnlstm_jaad_standardcrop --checkpoint_start_from None --restore_from_checkpoint 1 \
--use_gpu 1 --gpu_num 1 --timestep 3\

