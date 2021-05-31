# SAT_NEW
Primarily this code is available on https://github.com/yunjey/show-attend-and-tell

You have to follow all the required steps to setup this code with required libraries except the sequence to execute this code.

We just customised this code as per our own requirements with respect to the use of Open-I Indiana University Chest X-rays Dataset and also for following purposes

- Data Reading and Handeling
- Data Writing
- Parameters 
- Usage of number of images


Follow all the required steps given with original code to steup the enviornement


After that following steps to execute the code

- Copy source resized images (224,224) into "image/train2014_resized/" directory
- Download VGGNet19 model (link given with original code) into "data" directory 
- Run "Prepare_twoImages.py"
- Run "train.py"

Results would be stored in the respective split in data folder.

- 'all_score.csv' contains both candidate and reference reports along with computed metric scores.
- 'val.bleu.scores.txt' would contains accumulative metric scores.

Results can also be recomputed using recompute files.
