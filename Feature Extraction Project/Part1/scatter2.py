# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 16:33:31 2019

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

    M=m1+m2+m3
    Mu = (m1*mu1+m2*mu2+m3*mu3)/M

    term1 = mu1-Mu
    term2 = mu2-Mu
    term3 = mu3-Mu

    newTerm1 = m1*np.matmul(term1.T, term1)
    newTerm2 = m2*np.matmul(term2.T, term2)
    newTerm3 = m3*np.matmul(term3.T, term3)

    B = np.zeros((len(Y1[0]), len(Y1[0])))
    for i in range(0, len(Y1[0])):
        for j in range(0, len(Y1[0])):
            B[i, j] = newTerm1[i, j]+newTerm2[i, j]+newTerm3[i, j]

    covMat = np.matmul(B,B.T)
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
