# Clone github this

git clone https://github.com/joris-vaneyghen/Music-Source-Separation-Training.git

cd Music-Source-Separation-Training


# Build cocker image

bash
docker build -t train-mss .


# Run the container with secrets mounted:

docker run -it --gpus=all --rm \
  --name training-container \
  --mount type=bind,source="$(pwd)",target=/home/user/workdir \
  train-mss

# Check GPU's available
nvidia-smi
python check_gpu.py


# SET API KEYS

export KAGGLE_USERNAME=<your_kaggle_username>
export KAGGLE_KEY=<your_kaggle_api_key>

# Download pretrained model

wget -P checkpoints -O checkpoints/scnet_checkpoint_musdb18.ckpt https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v.1.0.6/scnet_checkpoint_musdb18.ckpt

# Download Datasets 

(these are private so use your own)

kaggle datasets download -d jorisvaneyghen/jazz-dataset-mss
kaggle datasets download -d jorisvaneyghen/jazz-extra-dataset-mss
kaggle datasets download -d jorisvaneyghen/jazz-extra-2-dataset-mss

unzip -d dataset jazz-dataset-mss.zip
unzip -d dataset jazz-extra-dataset-mss.zip
unzip -d dataset jazz-extra-2-dataset-mss.zip

rm *.zip

# Prepare data

python scripts/prepare_data_jazz_drums_piano_mono.py

# Train


python train.py \
    --model_type scnet \
    --config_path 'configs/JorisVaneyghen/config_jazz_mono_scnet.yaml' \
    --results_path checkpoints/ \
    --data_path 'dataset/train/dataset.csv' \
    --dataset_type 3 \
    --valid_path 'dataset/valid' \
    --num_workers 4 \
    --device_ids 0 1 2 3\
    --start_check_point 'checkpoints/scnet_checkpoint_musdb18.ckpt' \
    --pre_valid \
    --wandb_key <your_wandb_api_key>