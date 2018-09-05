我将原来的data1.csv里面的数据提取出来，删掉了没用的列，以及异常的片段；
然后分成三块(train[170000:470000], valid[0:60000], test[60000:120000])；
每块数据都是一个numpy的ndarray，都做了归一化，然后存储为三个npy对象文件；
读取数据只需使用numpy即可，例如 data_train = np.load('data_train.npy')