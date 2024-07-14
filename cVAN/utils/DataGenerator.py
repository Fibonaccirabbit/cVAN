import numpy as np


class kFoldGenerator():
    '''
    data Generator
    '''
    k = -1  # the fold number
    x_list = []  # x list with length=k
    y_list = []  # x list with length=k

    # Initializate
    def __init__(self, x, y):
        if len(x) != len(y):
            assert False, 'data generator: Length of x or y is not equal to k.'
        self.k = len(x)
        self.x_list = x
        self.y_list = y

    # Get i-th fold
    def getFold(self, i):
        isFirst = True

        for p in range(self.k):
            if p != i:
                if isFirst:
                    train_data = self.x_list[p]
                    train_targets = self.y_list[p]
                    isFirst = False
                else:
                    train_data = np.concatenate((train_data, self.x_list[p]))
                    train_targets = np.concatenate((train_targets, self.y_list[p]))
            else:
                val_data = self.x_list[p]
                # val_data[:, 2:9, :] = 0
                val_targets = self.y_list[p]
        return train_data, train_targets, val_data, val_targets

    # Get i-th fold
    def getBFold(self, i):
        isFirst = True
        isFirst2 = True
        for p in range(self.k):
            if p not in range(i, i + 10):
                if isFirst:
                    train_data = self.x_list[p]
                    train_targets = self.y_list[p]
                    isFirst = False
                else:
                    train_data = np.concatenate((train_data, self.x_list[p]))
                    train_targets = np.concatenate((train_targets, self.y_list[p]))
            else:
                if isFirst2:

                    val_data = self.x_list[p]
                    val_targets = self.y_list[p]
                    isFirst2 = False
                else:
                    val_data = np.concatenate((val_data, self.x_list[p]))
                    val_targets = np.concatenate((val_targets, self.y_list[p]))
        return train_data, train_targets, val_data, val_targets

    def getEvaluateFold(self, i):
        isFirst = True
        self.k = 10

        val_data = self.x_list[i]
        val_targets = self.y_list[i]
        return val_data, val_targets

    def getAllY(self):

        isFirst = True
        for p in range(self.k):
            if isFirst:
                train_targets = self.y_list[p]
                isFirst = False
            else:
                train_targets = np.concatenate((train_targets, self.y_list[p]))
        return train_targets

    def getTestFold(self, i):
        val_data = self.x_list[i]
        val_tragets = self.y_list[i]
        return val_data, val_tragets

    # Get all data x
    def getX(self):
        All_X = self.x_list[0]
        for i in range(1, self.k):
            All_X = np.append(All_X, self.x_list[i], axis=0)
        return All_X

    # Get all label y (one-hot)
    def getY(self):
        All_Y = self.y_list[0][2:-2]
        for i in range(1, self.k):
            All_Y = np.append(All_Y, self.y_list[i][2:-2], axis=0)
        return All_Y

    # Get all label y (int)
    def getY_int(self):
        All_Y = self.getY()
        return np.argmax(All_Y, axis=1)
