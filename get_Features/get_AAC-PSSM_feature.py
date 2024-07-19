import numpy as np
import csv


def gram1(pssm):
    feature1 = np.zeros(20)
    for i in range(20):
        feature1[i] = pssm[:, i].mean()
    feature1 = np.round(feature1, 6)
    feature = feature1.tolist()
    return feature


def gram2(pssm):
    feature2 = np.zeros(400)
    L = len(pssm)
    for j in range(20):
        for k in range(20):
            num = 0
            for i in range(L - 1):
                num += pssm[i, j] * pssm[i + 1, k]
            num /= (L - 1)
            index = 20 * j + k
            feature2[index] = num
    feature2 = np.round(feature2, 6)
    feature = feature2.tolist()
    return feature


def readpssm(dataset, filename):
    with open(f'../path/') as f:
        lines = f.readlines()[3:-6]
        pssm = np.array([line.split()[2:22] for line in lines], dtype=int)
    return pssm


def getpssmdata(seq_nums):
    classtarget = []
    feature = []

    for i in range(seq_nums):
        pssm = readpssm(str(i))
        feature1 = gram1(pssm)
        feature.append(feature1)
        classtarget.append(0)
    return feature, classtarget


def getfeature_gram1(seq_nums):
    feature, target = getpssmdata(seq_nums)
    strlen = len(feature)
    with open('../Features/Features_csv/name.csv', 'w', newline='') as file:
        content = csv.writer(file, dialect='excel')
        for i in range(strlen):
            content.writerow(feature[i])


def main():
    seq_nums = ''
    getfeature_gram1(seq_nums)


if __name__ == "__main__":
    main()