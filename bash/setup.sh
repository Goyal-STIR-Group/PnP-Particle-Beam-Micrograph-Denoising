# Extract Sponge dataset
cd data
tar -xvf spongeDataset.tar.gz
cd ..

# Download and extract DnCNN model weights
mkdir model
cd model
gdown https://drive.google.com/uc?id=1WkArJReKKGUX3TQIWzumnPch319C906m -O dncnn25.tar.gz
tar -xvf dncnn25.tar.gz
cp std25/checkpoints/dncnn-epoch\=69-val_psnr\=29.23.ckpt std25.ckpt
rm dncnn25.tar.gz
cd ..

# Make directory for saving results
mkdir result