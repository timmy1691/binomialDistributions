import numpy as np
import math

class Binomial:
    def __init__(self, desiredProb, numSamples, probBlock=None):
        """
        Input one probability, the over desired success probability 'DesiredProb"
        Or the success prob of a single block blockProb
        The number of samples in the binomial distribution numSamples
     
        """
        self.desiredProb = self.setProb(desiredProb)
        self.numSamples = numSamples
        self.probBlock = probBlock


    def setProb(self, desiredProb):
        if isinstance(desiredProb, float):
            if desiredProb < 1 and desiredProb > 0:
                return desiredProb
    
        raise ValueError("Desired probability must be a float between 0 and 1")
        


    def binomialSamples(self, n):
        """
        Binomail distribution probability for a fixed probability p and input sample size of n
        """
        if self.probBlock is not None:
            p = self.probBlock
        else:
            raise ValueError("Block probability must be set to calculate binomial distribution")
        minK = n//2 + 1
        totalProbability = 0
        for k in range(minK, n+1):
            totalProbability += math.comb(n, k) * (self.p ** k) * ((1 - self.p) ** (n - k))
        return totalProbability
    
    def binomialProb(self, p):
        """
        Calculate the total prob for binomial distribtuion for a fixed sample size n and success probability p
        """

        if p >= 1:
            raise ValueError("Probability must be less than 1")
        if p <= 0:
            raise ValueError("Probability must be greater than 0")
        
        numSamples = self.numSamples
        if numSamples % 2 == 0:
            minK = numSamples//2 + 1
        else:
            minK = (numSamples + 1)//2

        totalProbs = 0
        for i in range(minK, numSamples+1):
            totalProbs += math.comb(numSamples, i) * (p ** i) * ((1 - p) ** (numSamples - i))

        return totalProbs
    
    def messageCorrectness(self):
        """
        Message Correctness is given by receiving each block correctly.
        This means given a confideence in the message, one takes the nth root for n samples to get the confidence in each block.
        """
        if self.probBlock is None:
            confidence = self.desiredProb
            numSamples = self.numSamples
            self.probBlock = confidence**(1/numSamples)
            return self.probBlock

    
    def reverseBinomial(self):
        """
        Given a fixed sample, find the minimum probability to achieve confidence level q
        """
        q = self.desiredProb
        n = self.numSamples

        if n == 1:
            return q
        

        if q > 0.5:
            lowerP = 0.5
            upperP = 1.0
            while upperP - lowerP > 1e-6:
                midP = (lowerP + upperP) / 2
                if self.binomialProb(midP) <= q:
                    lowerP = midP
                else:
                    upperP = midP

        self.probBlock = midP    

        return midP


    def naiveMinSamples(self, q):
        """
        Naive method of finding the minimum number of samples to achieve confidence level q

        Calculate the analytical expected value
        """

        if self.probBlock <= 0.5:
            raise ValueError("Probability of success must be greater than 0.5")
        
        if q <= self.probBlock :
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