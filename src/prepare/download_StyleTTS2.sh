# working directory of StyleTTS2 and the pretrained models
FILE_LINK="https://surfdrive.surf.nl/s/9HcPQb8XcBDcsba"
DEST_DIR="../../checkpoints/"
mkdir -p $DEST_DIR

ZIP_FILE="./StyleTTS2.zip"

echo "Downloading working directory of StyleTTS2 and the pretrained models"
wget "$FILE_LINK/download" -O $ZIP_FILE

echo "Unzipping..."
unzip -q $ZIP_FILE -d $DEST_DIR
rm -f $ZIP_FILE
