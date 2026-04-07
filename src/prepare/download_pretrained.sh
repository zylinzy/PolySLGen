# Pertrained motion evaluator
FILE_LINK="https://surfdrive.surf.nl/s/fTq5T7nkAXEnAjG"
DEST_DIR="../../checkpoints/"
mkdir -p $DEST_DIR

ZIP_FILE="./motion_evaluator.zip"

echo "Downloading motion evaluator folder"
wget "$FILE_LINK/download" -O $ZIP_FILE

echo "Unzipping..."
unzip -q $ZIP_FILE -d $DEST_DIR
rm -f $ZIP_FILE

# Pretrained PolySLGen
FILE_LINK="https://surfdrive.surf.nl/s/TRoJE6L52G4eFDy"
DEST_DIR="../../"

ZIP_FILE="./pretrained_model.zip"

echo "Downloading pre-trained PolySLGen"
wget "$FILE_LINK/download" -O $ZIP_FILE

echo "Unzipping..."
unzip -q $ZIP_FILE -d $DEST_DIR
rm -f $ZIP_FILE