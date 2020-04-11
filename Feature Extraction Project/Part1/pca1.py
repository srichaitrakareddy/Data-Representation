# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 17:24:54 2019

@author: Sri Chaitra Kareddy
"""
import sys
import numpy as np
import csv

if __name__ == '__main__':
    if len(sys.argv) != 5 :
        print('usage : ', sys.argv[0], 'data_file labels_file output_vector_file reduced_data_file')
        sys.exit()
    Xt = np.genfromtxt(sys.argv[1], delimiter=',', dtype=float)
    y = np.genfromtxt(sys.argv[2],  delimiter=',', dtype=int)
    covMat = np.matmul(Xt.T, Xt)
    eigval, eigvec = np.linalg.eig(covMat)
    sortedIndices = np.argsort(eigval)[::-1]
    sortedEigenValues = eigval[sortedIndices]
    sortedEigenVector = eigvec[:, sortedIndices]
    feat1 = np.dot(Xt, sortedEigenVector[:,0].T)
    feat2 = np.dot(Xt, sortedEigenVector[:,1].T)
    with open(sys.argv[4], mode='a', newline='') as writeFile:
        writeFile = csv.writer(writeFile, delimiter=',')
        for i in range(0, len(feat1)):
            writeFile.writerow([feat1[i], feat2[i]])
    with open(sys.argv[3], mode='a', newline='') as writeFile:
        writeFile = csv.writer(writeFile, delimiter=',')
        writeFile.writerow(sortedEigenVector[:,0])
        writeFile.writerow(sortedEigenVector[:,1])
