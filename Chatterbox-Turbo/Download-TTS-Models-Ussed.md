## Navigate to your models directory
cd ~/chatterturbo/models/

## 1. Chatterbox-Turbo (350M) - Official Resemble AI Weights
wget https://huggingface.co/ResembleAI/chatterbox-turbo/resolve/main/t3_turbo_v1.safetensors -O chatterbox-turbo.safetensors

## 2. Chatterbox-Multilingual (500M) - Official Resemble AI Weights
wget https://huggingface.co/ResembleAI/chatterbox/resolve/main/t3_mtl23ls_v2.safetensors -O chatterbox-multilingual.safetensors

## 3. S3Gen Decoder (REQUIRED for both models to produce audio)
wget https://huggingface.co/ResembleAI/chatterbox-turbo/resolve/main/s3gen.safetensors -O s3gen.safetensors

## 4. Voice Encoder (REQUIRED for Zero-Shot Cloning)
wget https://huggingface.co/ResembleAI/chatterbox/resolve/main/ve.safetensors -O ve.safetensors
