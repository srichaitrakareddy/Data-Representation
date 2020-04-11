"""
@author: Sri Chaitra Kareddy
"""

import numpy as np
import math
import random
import sys
import csv

random.seed(1)
np.random.seed(1)

def assignClusters(Xt, Centers):
    k = Centers.shape[0]
    n = Xt.shape[0]
    m = Xt.shape[1]
    #find and store Euclidian distance of all points from the centers
    distanceMatrix = np.zeros((n, k))
    for j in range(0,n):
        for c in range(0,k):
            sqdist = 0
            for l in range(0, m):
                sqdist += math.pow((Xt[j][l] - Centers[c][l]),2)
            distanceMatrix[j][c] = sqdist
    #compare the distances and assign the clusters to eaxh of the points
    clusters = np.zeros((1,n))
    for j in range(0,n):
        c = np.argmin(distanceMatrix[j,:])
        clusters[0][j] = c
    return clusters


def recalculateCentroids(clusters, Xt):
    list1 = np.unique(clusters)
    k = list1.shape[0]
    m = Xt.shape[1]
    centers = np.zeros((k, m))
    for i in range(0,k):
        indices = np.where(clusters == i)
        newXt = Xt[indices[1],:]
        for j in range(0, m):
            centers[i][j] = np.mean(newXt[:,j])        
    return centers
    

def distanceFromCenters(Xt, centers, k):
    distanceMatrix = np.zeros((1,Xt.shape[0]))
    for i in range(0, Xt.shape[0]):
        distanceMatrix[0][i] = minDistanceFromAllPickedCenters(Xt[i,:], centers, k)
    
    return distanceMatrix


def minDistanceFromAllPickedCenters(X, centers, k):
    dist = np.zeros((1, k))
    for i in range(0,k):
        distPartialSum = 0
        for j in range(0, X.shape[0]):
            distPartialSum += math.pow((X[j]-centers[i][j]),2)
        dist[0][i] = math.sqrt(distPartialSum)
    
    return np.amin(dist)


if __name__ == '__main__':
    if len(sys.argv) != 5 :
        print('usage : ', sys.argv[0], 'data_file number_of_clusters sigma_vaue output_cluster_file')
        sys.exit()
    Xt = np.genfromtxt(sys.argv[1], delimiter=',', dtype=float) 
    k = int(sys.argv[2])
    sigma = float(sys.argv[3])
    n = Xt.shape[0]
    m = Xt.shape[1]
    pairwiseDistanceMatrix = np.zeros((n,n))
    for i in range(0, n):
        for j in range(0, n):
            dist = 0
            for l in range(0,m):
                dist += math.pow((Xt[i][l]-Xt[j][l]),2)
            pairwiseDistanceMatrix[i][j] = math.sqrt(dist)
    
    #calculating the weight matrix W
    W = np.zeros((n,n))
    for i in range(0,n):
        for j in range(0,n):
            W[i][j] = math.exp(-math.pow((pairwiseDistanceMatrix[i][j]),2)/(2*math.pow(sigma,2)))
      
    #calculating the degree matrix D    
    D = np.zeros((n,n))
    for i in range(0,n):
        weightSum = 0
        for j in range(0,n):
            weightSum += W[i][j]
        D[i][i] = weightSum
        
    #calculating the Lagrangian matrix L
    L = np.zeros((n,n))
    for i in range(0, n):
        for j in range(0, n):
            if(i==j and D[i][i] != 0):
                L[i][j] = 1
            elif(i!=j and D[i][i]!=0 and D[j][j]!=0):
                L[i][j] = -W[i][j]/(math.sqrt(D[i][i]*D[j][j]))
            else:
                L[i][j]=0
           
    eigval, eigvec = np.linalg.eig(L)
    sortedIndices = np.argsort(eigval)[::]
    sortedEigenValues = eigval[sortedIndices]
    sortedEigenVector = eigvec[:, sortedIndices]
    kEigenVector = sortedEigenVector[:,0:k]

    generalisedkEigenVectors = np.zeros((len(kEigenVector), k))
    for i in range(0,k):
        for j in range(0,len(kEigenVector)):
            generalisedkEigenVectors[j][i] = kEigenVector[j][i]/math.sqrt(D[j][j])
    
    #running kmeanspp on the vectors of the matrix L        
    Xt = generalisedkEigenVectors
    
    r = 10
    n = Xt.shape[0]
    m = Xt.shape[1]
    result = np.zeros((1,n))
    iterationAnswers = np.zeros((r, n))
    for i in range(0,r):
        Centers = np.zeros((k, m))
        l = random.randrange(0, n)
        Centers[0] = Xt[l]
        for j in range(1, k):
            distList = distanceFromCenters(Xt, Centers, j)
            distSum = distList.sum()           
            distList = distList/distSum
            indice = np.random.choice(n, p=distList.flatten())
            Centers[j] = Xt[indice]
            
        clusters = assignClusters(Xt, Centers)
        newCenters = recalculateCentroids(clusters, Xt)
        while (newCenters.any() != Centers.any()):
            Centers = newCenters
            clusters = assignClusters(Xt, Centers)
            newCenters = recalculateCentroids(clusters, Xt)
        iterationAnswers[i] = clusters
    iterationAnswersInt = iterationAnswers.astype(int,casting='unsafe', subok=True, copy=True)
    for i in range(0,n):
        y = np.bincount(iterationAnswersInt[:,i])
        ii = np.nonzero(y)[0]
        d = np.vstack((ii,y[ii])).T
        maxval = 0
        maxind = 0
        for j in range(0,d.shape[0]):
            if(maxval<d[j][1]):
                maxval = d[j][1]
                maxind = d[j][0]
        result[0][i] = maxind
        
    E = 0
    list1 = np.unique(result)
    k = list1.shape[0]
    m = Xt.shape[1]
    U_j = recalculateCentroids(result, Xt)
    for i in range(0,k):
        indices = np.where(result == i)
        newXt = Xt[indices[1],:]
        for j in range(0,newXt.shape[0]):
            for l in range(0, m):
                E += math.pow((newXt[j][l] - U_j[i][l]),2)
    
    print("Quantization Error is:")
    print(round(E, 4))
#    with open(sys.argv[4], mode='a', newline='') as writeFile:
#        writeFile = csv.writer(writeFile, delimiter=',')
#        writeFile.writerow(result)  
    
    with open(sys.argv[4], mode='w', newline='') as writeFile:
        writeFile = csv.writer(writeFile, delimiter=',')
        for i in range(0,n):
            writeFile.writerow([int(result[0][i])])        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    