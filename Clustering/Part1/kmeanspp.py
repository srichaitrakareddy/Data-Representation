"""
@author: Sri Chaitra Kareddy
"""
import sys
import numpy as np
import csv
import math
import random

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
        print('usage : ', sys.argv[0], 'data_file number_of_clusters number_of_iterations output_cluster_file')
        sys.exit()
    Xt = np.genfromtxt(sys.argv[1], delimiter=',', dtype=float)
    k = int(sys.argv[2])
    r = int(sys.argv[3])
    n = Xt.shape[0]
    m = Xt.shape[1]
    result = np.zeros((1,n))
    iterationAnswers = np.zeros((r, n))
    for i in range(0,r):
        Centers = np.zeros((k, m))
        #pick the first center randomly
        l = random.randrange(0, n)
        Centers[0] = Xt[l]
        #pick the next k-1 centers with a probability proportional to their distances from the picked centers
        for j in range(1, k):
            distList = distanceFromCenters(Xt, Centers, j)
            distSum = distList.sum()           
            distList = distList/distSum
            indice = np.random.choice(n, p=distList.flatten())
            Centers[j] = Xt[indice]
            
        clusters = assignClusters(Xt, Centers)
        newCenters = recalculateCentroids(clusters, Xt)
        #until clusters dont remain constant, keep reassigning clusters and recalculating centers
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