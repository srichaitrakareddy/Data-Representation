# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 19:11:56 2019

@author: Sri Chaitra Kareddy
"""
import csv
import sys
import numpy as np

if __name__ == '__main__':
    if len(sys.argv) != 5 :
        print('usage : ', sys.argv[0], 'data_file labels_file output_vector_file reduced_data_file')
        sys.exit()

    Xt = np.genfromtxt(sys.argv[1], delimiter=',', dtype=float)
    y = np.genfromtxt(sys.argv[2],  delimiter=',', dtype=int)

    indices = y==1
    Y1 = Xt[indices]

    indices = y==2
    Y2 = Xt[indices]

    indices = y==3
    Y3 = Xt[indices]

    mu1 = np.zeros((1,len(Y1[0])))
    mu2 = np.zeros((1,len(Y2[0])))
    mu3 = np.zeros((1,len(Y3[0])))

    m1 = 0
    m2 = 0
    m3 = 0

    for i1 in range(0, Y1.shape[0]):
        yi = Y1[i1]
        m1=m1+1
        for j in range(0, len(yi)):
            mu1[0,j] = mu1[0,j] + yi[j]
    for i1 in range(0, len(mu1)):
        mu1[i1] = mu1[i1]/m1

    for i1 in range(0, Y2.shape[0]):
        yi = Y2[i1]
        m2=m2+1
        for j in range(0, len(yi)):
            mu2[0,j] = mu2[0,j] + yi[j]
    for i1 in range(0, len(mu2)):
        mu2[i1] = mu2[i1]/m2

    for i1 in range(0, Y3.shape[0]):
        yi = Y3[i1]
        m3=m3+1
        for j in range(0, len(yi)):
            mu3[0,j] = mu3[0,j] + yi[j]
    for i1 in range(0, len(mu3)):
        mu3[i1] = mu3[i1]/m3

    S1 = np.zeros((Y1.shape[1], Y1.shape[1]))
    S2 = np.zeros((Y2.shape[1], Y2.shape[1]))
    S3 = np.zeros((Y3.shape[1], Y3.shape[1]))

    for i1 in range(0, Y1.shape[0]):
        yi = Y1[i1]
        term = yi.reshape((Y1.shape[1],1))-mu1.reshape((Y1.shape[1], 1))
        newTerm = np.matmul(term, term.T)
        for l in range(0, Y1.shape[1]):
            for k in range(0, Y1.shape[1]):
                S1[l][k] = S1[l][k]+newTerm[l][k]

    for i1 in range(0, Y2.shape[0]):
        yi = Y2[i1]
        term = yi.reshape((Y2.shape[1],1))-mu2.reshape((Y2.shape[1], 1))
        newTerm = np.matmul(term, term.T)
        for l in range(0, Y2.shape[1]):
            for k in range(0, Y2.shape[1]):
                S2[l][k] = S2[l][k]+newTerm[l][k]

    for i1 in range(0, Y3.shape[0]):
        yi = Y3[i1]
        term = yi.reshape((Y3.shape[1],1))-mu3.reshape((Y3.shape[1], 1))
        newTerm = np.matmul(term, term.T)
        for l in range(0, Y3.shape[1]):
            for k in range(0, Y3.shape[1]):
                S3[l][k] = S3[l][k]+newTerm[l][k]

    W = np.zeros((Xt.shape[1], Xt.shape[1]))

    for i in range(0, S1.shape[0]):
        for j in range(0, S1.shape[0]):
            W[i, j] =W[i, j]+S1[i, j]+S2[i, j]+S3[i, j]

    covMat = np.matmul(W,W.T)
    eigval, eigvec = np.linalg.eig(covMat)

    sortedIndices = np.argsort(eigval)[::]
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
