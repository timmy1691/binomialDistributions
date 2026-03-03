import numpy as np
from collections import Counter
import math

class Multinomial:
    def __init__(self, p, numCats = None, desiredProb = None):
        self.setProbs(p)
        self.numCats = len(p)

    def setProbs(self, p):
        if isinstance(p, list):
            self.p = max(p)
            self.index = p.index(max(p))
            self.q = (1 - self.p) / (len(p) - 1)
        else:
            self.p = p
            self.q = (1 - p) / (self.numCats - 1)

    def simulatedMultinomialSampling(self, numSamples, probVector, desiredIndex, numTrials=1000):
        """
        Sampling for emperical estimation of number of samples to form shape of distribution.
        input: probability vector
        """

        successCount = 0

        for _ in range(numTrials):
            samples = np.random.choice(range(self.numCats), size=numSamples, p=probVector, replace=True)
            counts = Counter(samples)
            most_common = counts.most_common(2)
            if len(most_common) < 2:
                # unique count from the sampler
                if most_common[0][0] == desiredIndex:
                    successCount += 1
                
            else:
                # if two things are unique count
                max_count = most_common[0][1]
                second_max_count = most_common[1][1]
                if most_common[0][0] == desiredIndex and max_count > second_max_count:
                    successCount += 1
                    
        return successCount / numTrials
    

    def minSampleFinderSimulation(self, q, desiredIndex = None, numTrials=1000):
        """
        Simulated method of finding minimum number of samples to achieve confidence level q

        brute force find the emperical sample size
        """

        if desiredIndex is None:
            probVector = [self.p] + [self.q] * (self.numCats - 1)
            desiredIndex = 0
        else:
            if desiredIndex >= self.numCats:
                raise ValueError("Desired index must be one of the categories")
            else:
                probVector = [self.q]*desiredIndex + [self.p] + [self.q]*(self.numCats - desiredIndex - 1)
            
        numSamples = 1
        samplesRes = {}
        while True:
            successRate = self.simulatedMultinomialSampling(numSamples, probVector, desiredIndex, numTrials=numTrials)
            samplesRes[numSamples] = successRate
            if successRate >= q:
                return numSamples
            numSamples += 1

        



    def binning(self, numSamples):
        """
        Number of samples is clear majority, then need to distribute the remaining samples
        Number of ways to permute is given by (n + k - 1) C (k-1)
        """    
        numCats = self.numCats - 1
        return math.comb(numSamples - 1, numCats - 1)


    def violations(self, numSamples, maxCount):
        """
        Count the number of violating setups
        """
        numCats = self.numCats
        
        maxViolations = numSamples //maxCount 
        if maxViolations >= numCats:
            return -1
        violationCount  = 0
        for i in range(1, maxViolations + 1):
            violationCount += math.comb(numCats, i) * (-1)**(i+1) * math.comb(numSamples-i*(maxCount) + numCats - 1, numCats - 1)
        
        return violationCount
    
