
# unzip dataset zip files
unzip data/test/processed/test-processed.zip -d data/test/processed
unzip data/train/processed/train-processed.zip -d data/train/processed

GREEN='\033[1;32m'
RED='\033[1;31m'
NC='\033[0m'

if [ $? -eq 0 ]; then
    printf "${GREEN}Datasets extracted successfully.${NC}"
else
    printf "${RED}Dataset extraction failed.${NC}"
fi
