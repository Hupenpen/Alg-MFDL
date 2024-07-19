import numpy as np
import csv

def gram1(pssm):
    feature1 = np.zeros(20)
    for i in range(20):
        feature1[i] = pssm[:, i].mean()
    feature1 = np.round(feature1, 6)
    return feature1.tolist()

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
    return feature2.tolist()

def readpssm(dataset, filename):
    with open(f'../path/{dataset}/{filename}') as f:
        lines = f.readlines()[3:-6]
        pssm = np.array([line.split()[2:22] for line in lines], dtype=int)
    return pssm

def getpssmdata(seq_nums):
    features = []
    classtarget = []

    for i in range(seq_nums):
        pssm = readpssm(str(i))
        feature = gram2(pssm)
        features.append(feature)
        classtarget.append(0)
    return features, classtarget

def getfeature_gram2(seq_nums):
    features, _ = getpssmdata(seq_nums)
    with open('../path/Features_csv/name.csv', 'w', newline='') as file:
        writer = csv.writer(file, dialect='excel')
        writer.writerows(features)

def main():
    getfeature_gram2()

if __name__ == "__main__":
    main()