docker run -it --gpus all \
  --name merlin \
  -p 8000:8000 \
  -p 8002:8002 \
  -p 8003:8003 \
  -p 8888:8888 \
  -v /home/kmanasu/PxC-book-recommendation/data-prep-EDA/clean:/workspace/data/ \
  -v /home/kmanasu/PxC-book-recommendation/DLRM/:/workspace/ \
  --ipc=host \
  nvcr.io/nvidia/merlin/merlin-tensorflow:nightly \
  /bin/bash
