MODEL_PATH="google/gemma-3-27b-it"

export CUDA_VISIBLE_DEVICES=4,5,6,7
vllm serve $MODEL_PATH \
  --host 0.0.0.0 \
  --port 8001 \
  --max-num-seqs 64 \
  --max-model-len 52768 \
  -tp 4 \
  -pp 1 \
  --gpu-memory-utilization 0.97
#   --enable-prefix-caching
