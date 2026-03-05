import numpy as np
from collections import Counter
import math

class Multinomial:
    def __init__(self, p, numCats = None, desiredProb = None):
        self.setProbs(p, numCats=numCats)
        # self.numCats = len(p)
        self.desiredProb = desiredProb

    def setProbs(self, p, numCats):
        if isinstance(p, list):
            self.p = max(p)
            self.index = p.index(max(p))
            self.q = (1 - self.p) / (len(p) - 1)
            self.numCats= len(p)
        else:
        
            if numCats is None:
                raise ValueError("number of categories needs to be defined")
            self.numCats = numCats
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
                return samplesRes
            numSamples += 1

        
    # def minSampleFinder(self, correctnessLevel):
    #     """
    #     Find the minimal number of samples to guarantee pluraiklty of the target with a certain confidence level q

    #     """

    #     numCats = self.numCats
    #     if correctnessLevel <= self.p:
    #         return 1
        
    #     if correctnessLevel > 0.5 and self.p < self.q:
    #         raise ValueError("Desired confidence level cannot be achieved with given probabilities")
        

    #     analyticalBound = 0
    #     numSamples = 2

    #     while analyticalBound < correctnessLevel:
    #         currentBound = 0
    #         minPlurality = math.ceil(numSamples / numCats) + 1
    #         majority = math.ceil(numSamples / 2) + 1

    #         if minPlurality == numSamples:
    #             currentBound = self.p ** numSamples
    #             numSamples += 1
    #             analyticalBound = currentBound
    #             continue

            
    #         for number in range(minPlurality, numSamples + 1):
    #             remainingSamples = numSamples - number
    #             tempVal = math.factorial(numSamples) / math.factorial(number) * self.p ** number
    #             maxCount = number - 1

    #         analyticalBound = currentBound
    #         numSamples += 1



    def permuter(self, numSamples):
        from itertools import combinations_with_replacement
        numCats = self.numCats - 1
        for thing in combinations_with_replacement(range(numCats), numSamples):           
            yield thing



    def computePermCoef(self, permutation, maxCount):
        """
        Permutation input will be a counter
        if maxCoun
        """

        from collections import Counter
        permCount = Counter(permutation)

        if max(permCount.values()) >= maxCount:
            return 0
        

        numCats = self.numCats - 1

        finalCounts = 1 
        for val in permutation:
            finalCounts *= math.factorial(permutation[val])
    
        return 1/ finalCounts


    def probCalculator(self, numSamples):
        """
        Probability of getting a plurality vote 
        Brute force go through all probabilities for a single number of samples, and the success probability fixed
        """
        numCats = self.numCats
        
        if numSamples % 2 == 0:
            majority = numSamples // 2 + 1
        else:
            majority = math.ceil(numSamples / 2)

        
        minPlurality = math.ceil(numSamples / numCats) + 1

        totalProb = 0
        for n in range(minPlurality, numSamples):
            if n > majority:
                intCoef = (math.factorial(numSamples) / math.factorial(n))
                tempCoef = self.p**(n) * self.q**(numSamples - n)

                permutationCoef = 0
                for perm in self.permuter(numSamples - n):
                    permutationCoef += self.computePermCoef(perm, n)


                intCoef = intCoef*permutationCoef
                totalProb += intCoef*tempCoef

        return totalProb
    

    def minSampleCalculator(self, q):
        """
        Find the min samples necesary for a plularity level of q
        Inopuyt q
        """

        bigNumber = 10**6
        
        while True:
            tempUpperBound = self.probCalculator(bigNumber)
            if tempUpperBound > q:
                upperBound = bigNumber
                break
            if tempUpperBound == q:
                return bigNumber
            else:
                lowerBound = bigNumber
                bigNumber = bigNumber * 2

        while lowerBound != upperBound:
            midVal = math.ceil((lowerBound + upperBound) / 2)
            currentProb = self.probCalculator(midVal)

            if currentProb > q:
                upperBound = midVal

            elif currentProb < q:
                lowerBound = midVal
            
            else:
                return midVal
                        

        

        








        
    def binning(self, numSamples):
        """
        Number of samples is clear majority, then need to distribute the remaining samples
        Number of ways to permute is given by (n - 1) C (k-1)
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
    
