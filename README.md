# An-implementation-of-Search-based-User-Interest-Model
Based on the Taobao dataset, the SIM model trained by pointwise is reproduced, most of the parameters are based on the paper, and the rest is a reasonable supplement to the project authors

# Training
1.First, get the Taobao dataset(https://tianchi.aliyun.com/dataset/649) and sample a certain interaction as a csv file (you can also use the entire dataset)

2.Run cutcsv.py and then data_split.py on your data to get a dictionary of user behavior sequences, as well as a test and training set

3.Modify the parameters and run the train.py

