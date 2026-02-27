import numpy as np
import math
from matplotlib import pyplot as plt

class Binomial:
    def __init__(self, p):
        self.p = p



    def binomial(self, n):
        """
        Binomail distribution to find the number of samples with success probability and result probability q
        """

        minK = n//2 + 1
        totalProbability = 0
        for k in range(minK, n+1):
            totalProbability += math.comb(n, k) * (self.p ** k) * ((1 - self.p) ** (n - k))
        
        return totalProbability


    def naiveMinSamples(self, q):
        """
        Naive method of finding the minimum number of samples to achieve confidence level q

        Calculate the analytical expected value
        """

        if self.p <= 0.5:
            raise ValueError("Probability of success must be greater than 0.5")
        
        if q <= self.p :
            return 1

        currentProb = 0
        currentN = 1
        while currentProb < q:
            resProb = self.binomial(currentN)
            if resProb >= q:
                return currentN
            else:
                currentN += 1




    def minSamples(self, q):
        """
        Is there a more clever way than naively searching?
        """

        pass


    def simulatedSampling(self, numSamples, numTrials=1000):
        """
        Sampling for emperical estimation of number of samples to form shape of distribution.
        input: probability vector
        """
        successCount = 0

        for i in range(numTrials):
            samples = np.random.choice(2, size=numSamples, p=[1-self.p, self.p])
            successCount += np.sum(samples)

        return successCount/numTrials
    

    def minSamplesSimulated(self, q, numTrials=1000):
        """
        Simulated method of finding minimum number of samples to achieve confidence level q
        """

        if self.p <= 0.5:
            raise ValueError("Probability of success must be greater than 0.5")
        
        if q <= self.p :
            return 1

        currentProb = 0
        currentN = 1
        resColl = {}
        while currentProb < q:
            resProb = self.simulatedSampling(currentN, numTrials)
            resColl[currentN] = resProb
            if resProb >= q:
                return currentN
            else:
                currentN += 1