# Pre-processed DnD dataset
FILE_LINK="https://surfdrive.surf.nl/s/rAo9aL2PcDrpdGL"
DEST_DIR="../../"

ZIP_FILE="./dnd_dataset_preprocessed.zip"

echo "Downloading the pre-processed DnD Group Gesture Dataset"
wget "$FILE_LINK/download" -O $ZIP_FILE

echo "Unzipping..."
unzip -q $ZIP_FILE -d $DEST_DIR
rm -f $ZIP_FILE
