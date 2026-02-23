# Chatter box 21GB
# Slim version do not contain voice models 6GB only

# 1. Create a clean local directory
cd /app/
mkdir -p models/chatterbox
mkdir -p models/chatterbox-turbo

# 2. Use the huggingface-cli to download the REAL files (no symlinks)
# Download the Standard/Multilingual weights
huggingface-cli download ResembleAI/chatterbox --local-dir ./models/chatterbox --local-dir-use-symlinks False
# Download the Turbo weights
huggingface-cli download ResembleAI/chatterbox-turbo --local-dir ./models/chatterbox-turbo --local-dir-use-symlinks False

# Check the files and size
du -sh ./models/*
