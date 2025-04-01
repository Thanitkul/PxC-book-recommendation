export CUDA_VISIBLE_DEVICES=1
export VLLM_USE_V1=0
MODEL_PATH="google-bert/bert-base-uncased"
# Start vLLM server
vllm serve "$MODEL_PATH" \
  -tp 1 \
  -pp 1 \
  --host 0.0.0.0 \
  --port 8082 \
  --max-num-seqs 256 \
  --gpu-memory-utilization 0.95
