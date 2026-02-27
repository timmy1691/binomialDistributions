import numpy as np
from collections import Counter
from matplotlib import pyplot as plt
import math

class Multinomial:
    def __init__(self, p):
        self.p = p

    
    def simulatedMultinomialSampling(self, numSamples,desiredIndex = None, numTrials=1000):
        """
        Sampling for emperical estimation of number of samples to form shape of distribution.
        input: probability vector
        """

        successCount = 0
        if desiredIndex is None:
            max_p = self.p.index(max(self.p))
        else:
            max_p = desiredIndex


        for i in range(numTrials):
            samples = np.random.choice(range(self.p), size=numSamples, p=self.p)
            counts = Counter(samples)
            most_common = counts.most_common(1)
            if len(most_common) < 2:
                if len(most_common) == 1:
                    if most_common[0][0] == max_p:
                        successCount += 1
                    continue

            max_count = most_common[0][1]
            second_max_count = most_common[1][1]
            difference = max_count - second_max_count
            if difference > 0:
                successCount += 1
        
        return successCount / numTrials
    

    def minSampleFinderSimulation(self, q,desiredIndex = None, numTrials=1000):
        """
        Simulated method of finding minimum number of samples to achieve confidence level q
        """

        if desiredIndex is None:
            desiredIndex = self.p.index(max(self.p))

        numSamples = 1
        samplesRes = {}
        while True:
            successRate = self.simulatedMultinomialSampling(numSamples, desiredIndex=desiredIndex, numTrials=numTrials)
            samplesRes[numSamples] = successRate
            if successRate >= q:
                return numSamples
            numSamples += 1


