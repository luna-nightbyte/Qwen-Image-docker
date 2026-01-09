docker run \
  --name qwen-image-edit-demo \
  --gpus '"device=0"' \
  --shm-size 24g \
  --restart unless-stopped \
  -p 7861:7861 \
  -e OPTIMIZATION_MODE=quantized \
  -e MODEL_PATH=/app/models/Qwen-Image-Edit-2511} \
  -e CUDA_VISIBLE_DEVICES=0 \
  -e GRADIO_SERVER_NAME=0.0.0.0 \
  -e GRADIO_SERVER_PORT=7861 \
  -e PYTORCH_ALLOC_CONF=expandable_segments:True \
  -e ENABLE_CPU_OFFLOAD=true \
  -e LOW_VRAM_MODE=false \
  -v qwen-huggingface-cache:/root/.cache/huggingface \
  -v ./models:/app/models \
  -v ./outputs:/app/outputs \
  lunanightbyte/qwen-image:latest \
  python src/examples/edit_demo.py