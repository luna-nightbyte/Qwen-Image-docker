# Docker Setup for Qwen-Image

This guide provides instructions for running Qwen-Image using Docker and Docker Compose.

> **🚀 For offline setup with local models, see [QUICKSTART_OFFLINE.md](QUICKSTART_OFFLINE.md)**
> **📖 For detailed offline guide, see [OFFLINE_SETUP.md](OFFLINE_SETUP.md)**

## Prerequisites

1. **Docker**: Install Docker (version 20.10 or higher)
   ```bash
   curl -fsSL https://get.docker.com -o get-docker.sh
   sudo sh get-docker.sh
   ```

2. **NVIDIA Docker Runtime**: Required for GPU support
   ```bash
   # Add NVIDIA package repositories
   distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
   curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
   curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

   # Install nvidia-docker2
   sudo apt-get update
   sudo apt-get install -y nvidia-docker2

   # Restart Docker daemon
   sudo systemctl restart docker
   ```

3. **Docker Compose**: Install Docker Compose (version 1.29 or higher)
   ```bash
   sudo apt-get install docker-compose-plugin
   ```

## Quick Start

### 1. Configure Environment Variables

Copy the example environment file and edit it with your settings:

```bash
cp .env.example .env
nano .env  # Edit with your settings
```

**For Offline Mode (Recommended):**
- Leave `DASHSCOPE_API_KEY` empty
- Ensure `MODEL_PATH=/app/models/Qwen-Image` points to your local model
- See [OFFLINE_SETUP.md](OFFLINE_SETUP.md) for downloading models

**For Online Mode (with prompt enhancement):**
- Set `DASHSCOPE_API_KEY` to your API key from https://dashscope.console.aliyun.com/
- Models will be downloaded automatically from HuggingFace on first run

### 2. Build and Run

#### Option A: Using Docker Compose (Recommended)

Build and start the service:
```bash
docker-compose up -d
```

View logs:
```bash
docker-compose logs -f qwen-image
```

Stop the service:
```bash
docker-compose down
```

#### Option B: Using Docker directly

Build the image:
```bash
docker build -t qwen-image:latest .
```

Run the container:
```bash
docker run -d \
  --name qwen-image-demo \
  --gpus all \
  -p 7860:7860 \
  --shm-size 8g \
  -e DASHSCOPE_API_KEY=your-api-key-here \
  -e NUM_GPUS_TO_USE=1 \
  -v huggingface-cache:/root/.cache/huggingface \
  qwen-image:latest
```

### 3. Access the Web Interface

Open your browser and navigate to:
```
http://localhost:7860
```

## Advanced Configuration

### Multi-GPU Setup

To use multiple GPUs, modify your `.env` file:
```bash
NUM_GPUS_TO_USE=4
CUDA_VISIBLE_DEVICES=0,1,2,3
```

Or specify in docker-compose:
```bash
docker-compose up -d --scale qwen-image=1
```

### Running the Edit Demo

To run the image editing demo instead:
```bash
docker-compose --profile edit up -d qwen-image-edit
```

Access at: `http://localhost:7861`

### Custom Model Paths

If you have pre-downloaded models, mount them as volumes:
```yaml
volumes:
  - /path/to/your/models:/root/.cache/huggingface
```

### Development Mode

For development with hot-reload, the `docker-compose.yml` already mounts the `./src` directory. Any changes to Python files will be reflected after restarting:
```bash
docker-compose restart qwen-image
```

## GPU Requirements

- **Minimum**: 1x NVIDIA GPU with 16GB VRAM
- **Recommended**: 1x NVIDIA GPU with 24GB+ VRAM (for faster inference)
- **Multi-GPU**: 2-4 GPUs for concurrent request processing

Supported GPU architectures:
- Volta (V100)
- Turing (T4, RTX 20xx)
- Ampere (A100, RTX 30xx)
- Ada Lovelace (RTX 40xx)
- Hopper (H100)

## Resource Configuration

### Memory Settings

Adjust shared memory size for larger batches:
```yaml
shm_size: '16gb'  # Increase if needed
```

### Queue Settings

Configure task queue size and timeout in `.env`:
```bash
TASK_QUEUE_SIZE=200  # Maximum queued tasks
TASK_TIMEOUT=600     # Task timeout in seconds
```

## Troubleshooting

### GPU Not Detected

Check if NVIDIA runtime is properly configured:
```bash
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

### Out of Memory Errors

1. Reduce `NUM_GPUS_TO_USE` or `TASK_QUEUE_SIZE`
2. Increase `shm_size` in docker-compose.yml
3. Use a GPU with more VRAM

### Port Already in Use

Change the port mapping in docker-compose.yml:
```yaml
ports:
  - "8080:7860"  # Map to port 8080 instead
```

### Models Not Downloading

Ensure you have internet connection and enough disk space. Models are cached in the Docker volume `huggingface-cache`.

To manually clear cache:
```bash
docker volume rm qwen-image_huggingface-cache
```

## Performance Optimization

### Using bfloat16 (Recommended)

The Dockerfile already uses bfloat16 by default for faster inference with lower memory usage.

### Caching Models

Models are automatically cached in a Docker volume. First run will download ~40GB of model weights.

### Production Deployment

For production, consider:
1. Using nginx as a reverse proxy
2. Enabling HTTPS
3. Setting up proper logging and monitoring
4. Using orchestration tools like Kubernetes

## Container Management

### View Running Containers
```bash
docker-compose ps
```

### Check Container Logs
```bash
docker-compose logs -f
```

### Enter Container Shell
```bash
docker-compose exec qwen-image bash
```

### Update to Latest Code
```bash
git pull
docker-compose build --no-cache
docker-compose up -d
```

### Clean Up Everything
```bash
docker-compose down -v  # Removes volumes too
docker system prune -a  # Clean up unused images
```

## Security Notes

1. Never commit your `.env` file with real API keys
2. Use environment variables for sensitive data
3. Consider using Docker secrets for production
4. Limit network exposure using firewall rules
5. Keep Docker and NVIDIA drivers updated

## Support

For issues related to:
- **Qwen-Image**: See main [README.md](README.md)
- **Docker setup**: Check this document or open an issue
- **GPU support**: Refer to [NVIDIA Docker documentation](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

## License

Same as the main project - Apache 2.0
