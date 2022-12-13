# Multi-node DDP
python main.py train configs/mks/mks000.yaml --num-workers 4 \
    --gpus 1 --num_nodes 2 --accelerator ddp --precision 16 \
    --benchmark --sync_batchnorm --kfold 0

# Single-node DDP
python main.py train configs/mks/mk000.yaml --num-workers 4 \
    --gpus 2 --num_nodes 1 --strategy ddp --precision 16 \
    --benchmark --sync_batchnorm --kfold 0

# Single GPU
python main.py train configs/mks/mk000.yaml --num-workers 4 \
    --gpus 1 --num_nodes 1 --precision 16 \
    --benchmark --kfold 0
