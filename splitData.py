import os
import random
import shutil
from itertools import islice

outputFolderPath = "Dataset/SplitData"
inputFolderPath = "Dataset/all"
splitRatio = {"train": 0.7, "val": 0.2, "test": 0.1}
classes = ["fake", "real"]
processedFile = "Dataset/processed_images.txt"

# Create the output directory and subdirectories
os.makedirs(f"{outputFolderPath}/train/images", exist_ok=True)
os.makedirs(f"{outputFolderPath}/train/labels", exist_ok=True)
os.makedirs(f"{outputFolderPath}/val/images", exist_ok=True)
os.makedirs(f"{outputFolderPath}/val/labels", exist_ok=True)
os.makedirs(f"{outputFolderPath}/test/images", exist_ok=True)
os.makedirs(f"{outputFolderPath}/test/labels", exist_ok=True)

# Load already processed images
if os.path.exists(processedFile):
    with open(processedFile, 'r') as f:
        processed_images = set(line.strip() for line in f)
else:
    processed_images = set()

# Get the list of image filenames without extensions
listNames = os.listdir(inputFolderPath)
uniqueNames = [name.split('.')[0] for name in listNames if name.lower().endswith('.jpg')]
uniqueNames = list(set(uniqueNames))

# Check if there are any images to process
if not uniqueNames:
    print("No images found in the input directory.")
    exit(1)

# Filter out already processed images
newImages = [name for name in uniqueNames if name not in processed_images]

# Check if there are new images to process
if not newImages:
    print("No new images to process.")
    exit(1)

# Shuffle the filenames
random.shuffle(newImages)

# Calculate the number of images for each split
lenData = len(newImages)
lenTrain = int(lenData * splitRatio['train'])
lenVal = int(lenData * splitRatio['val'])
lenTest = int(lenData * splitRatio['test'])

# Adjust the training size if necessary
if lenData != lenTrain + lenTest + lenVal:
    remaining = lenData - (lenTrain + lenTest + lenVal)
    lenTrain += remaining  # Add remaining images to the training set

# Split the list of filenames
lengthToSplit = [lenTrain, lenVal, lenTest]
Input = iter(newImages)
Output = [list(islice(Input, elem)) for elem in lengthToSplit]
print(f'Total Images: {lenData} \nSplit: {len(Output[0])} {len(Output[1])} {len(Output[2])}')

# Copy the files into the appropriate directories
sequence = ['train', 'val', 'test']
for i, out in enumerate(Output):
    for fileName in out:
        imagePath = f'{inputFolderPath}/{fileName}.jpg'
        labelPath = f'{inputFolderPath}/{fileName}.txt'

        # Check if the image and label exist before copying
        if os.path.exists(imagePath):
            shutil.copy(imagePath, f'{outputFolderPath}/{sequence[i]}/images/{fileName}.jpg')
        if os.path.exists(labelPath):
            shutil.copy(labelPath, f'{outputFolderPath}/{sequence[i]}/labels/{fileName}.txt')

        # Record processed images
        processed_images.add(fileName)

# Save updated processed images list
with open(processedFile, 'w') as f:
    for image in processed_images:
        f.write(f"{image}\n")

print("Split process completed.....")

# Create the Data.yaml file
dataYaml = f'path: ../Data\n\
train: ../train/images\n\
val: ../val/images\n\
test: ../test/images\n\
\n\
nc: {len(classes)}\n\
names: {classes}'

with open(f"{outputFolderPath}/data.yaml", 'w') as f:
    f.write(dataYaml)

print("Data.yaml file Created.......")
