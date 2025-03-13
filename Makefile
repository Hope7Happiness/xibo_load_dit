sample:
	python sample.py --image-size 512 --seed 1

convert:
#	conda activate NNX
	python load_from_jax.py --path ./jax_models/20250311_203603_y8zcr8_kmh-tpuvm-v2-32-1__b_lr_ep_eval_checkpoint_384000/

run:
	python fm_sample.py --seed 1 --model DiT-B/4 --num-classes 1000 --ckpt ./torch_models/20250311_203603_y8zcr8_kmh-tpuvm-v2-32-1__b_lr_ep_eval_checkpoint_384000.pt