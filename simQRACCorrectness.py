import numpy as np
from collections import Counter
import math
from tqdm import tqdm
import multiprocessing as mp
from multiprocessing import Pool

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
                if most_common[0][0] == desiredIndex and max_count != second_max_count:
                    successCount += 1
                    
        return successCount / numTrials
    

    def simulatedMultinomailMajoritySampling(self, numSamples, probVector, desiredIndex, numTrials=1000):
        """
        Monte Carlo simulation for majority sampling, for results that are strictly greater than half
        """
        if numSamples % 2 == 0:
            majority = numSamples // 2 + 1
        else:
            majority = (numSamples + 1) // 2

        successes = 0
        for _ in numTrials:
            samples = np.random.choice(range(self.numCats), size=numSamples, p=probVector, replace=True)
            counts = Counter(samples)
            mostCommon = counts.most_common(1)
            if mostCommon[0][0] == desiredIndex and mostCommon[0][1] >= majority:
                successes += 1
        return successes / numTrials
    

    def singleTrial(self, probvector, numSamples, desiredIndex, trialIndex = 0):
        """
        Perform single monte carlo simulation for a fixed numSamples
        """
        if desiredIndex is None:
            desiredIndex = probvector.index(max(probvector))
        else:
            if probvector.index(max(probvector)) != desiredIndex:
                raise ValueError("the value of the desired index must be the same as the biased prob vector")
            
        samples = np.random.choice(range(self.numCats), size=numSamples, p=probvector)
        counts = Counter(samples)
        most_common = counts.most_common(2)
        if len(most_common) < 2:
            # unique count from the sampler
            if most_common[0][0] == desiredIndex:
                return 1
        else:
            # if two things are unique count
            max_count = most_common[0][1]
            second_max_count = most_common[1][1]
            if most_common[0][0] == desiredIndex and max_count != second_max_count:
                return 1
        return 0


    def multiprocessingMCPLuralitySim(self, numSamples, probVector, desiredIndex, numPools=5, numTrials=1000):
        """
        Monte Carlo method of plurality simulation, multiprocessing
        """
        simRes = []
        from functools import partial
        parametrizedFunction = partial(self.singleTrial, numSamples = numSamples, desiredIndex=desiredIndex, probVector=probVector)
        with mp.Pool(processes=numPools) as pool:
            simRes.append(pool.imap_unordered(parametrizedFunction, self.trialPermuter(numTrials)))

        return sum(simRes) / numTrials
    

    def singleSampleSimulated(self, numSamples, desiredIndex= None, numTrials=1000):
        if desiredIndex is None:
            probVector = [self.p] + [self.q] * (self.numCats - 1)
            desiredIndex = 0
        else:
            if desiredIndex >= self.numCats:
                raise ValueError("Desired index must be one of the categories")
            else:
                probVector = [self.q]*desiredIndex + [self.p] + [self.q]*(self.numCats - desiredIndex - 1)

        prob = self.simulatedMultinomialSampling(numSamples, probVector, desiredIndex, numTrials)
        return prob


    def minSampleFinderSimulationLinear(self, q, desiredIndex= None, startSample=None, numTrials=1000):
        """
        Perform the sampling with linear indcreasing numOfSamples
        
        """
        if desiredIndex is None:
            probVector = [self.p] + [self.q] * (self.numCats - 1)
            desiredIndex = 0
        else:
            if desiredIndex >= self.numCats:
                raise ValueError("Desired index must be one of the categories")
            else:
                probVector = [self.q]*desiredIndex + [self.p] + [self.q]*(self.numCats - desiredIndex - 1)


        if startSample is None:
            numSamples = 3
        else:
            if isinstance(startSample, int):
                numSamples = startSample
            else:
                raise ValueError("sample number must be positive integer")
            
        allSampleProbs = {}
        while True:
            tempProb = self.simulatedMultinomialSampling(numSamples=numSamples, probVector= probVector, desiredIndex=desiredIndex, numTrials=numTrials)
            allSampleProbs[numSamples] = tempProb
            if tempProb > q:
                break
            numSamples += 1
        
        return allSampleProbs, numSamples


    def minSampleFinderSimulationBinSearch(self, q, desiredIndex = None, numTrials=1000):
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
            
        numSamplesLowerBound = 1
        numSamplesUpperBound = 10000

        while True:
            successRate = self.simulatedMultinomialSampling(numSamplesUpperBound, probVector, desiredIndex, numTrials=numTrials)
            if successRate > q:
                break
            else:
                numSamplesLowerBound = numSamplesUpperBound
                numSamplesUpperBound = numSamplesUpperBound * 2


        samplesRes = {}
        while numSamplesUpperBound != numSamplesLowerBound:
            # print("lower bound")
            midPoint = math.ceil((numSamplesLowerBound + numSamplesUpperBound)/2) + 1
            successRate = self.simulatedMultinomialSampling(midPoint, probVector, desiredIndex, numTrials=numTrials)
            samplesRes[midPoint] = successRate
            if successRate < q:
                numSamplesLowerBound = midPoint
            else:
                numSamplesUpperBound = midPoint
        return samplesRes, midPoint
    


    def trialPermuter(self, numTrials):
        for i in range(numTrials):
            yield i


    def permuter(self, numSamples, maxCount):
        """
        The subset needed is defined by maxcount of the bias index
        numSamples is the total distirbuted samples

        maxCount is the maxCount of the biased index
        numSamples is the number of samples in total to be distributed
        """
        from itertools import product
        numCats = self.numCats - 1
        minConsideredValues = min(numSamples, maxCount)
        for thing in product(range(minConsideredValues), repeat=numCats):
            if sum(thing) == numSamples:
                yield thing

    def noGuardPermuter(self, numSamples):

        from itertools import product
        numCats = self.numCats - 1
        for thing in product(range(numSamples), repeat=numCats):
            if sum(thing) == numSamples:
                yield thing

    def computeRamanApproxSinglePermExp(self, permutation):
        """
        compute producte of fatcorials using ramanujan approx

        compute log(prod(xi!)) = (-1) sum(log(x!))
        """
        from distributions.helperFunctions.multinomailApprox import ramanujanApproxNumpy

        loggedVals = ramanujanApproxNumpy(permutation)
        summedVals = loggedVals.sum()
        return math.exp(-summedVals)
    
    def computeRamanApproxSinglePerm(self, permutation):
        """
        compute producte of fatcorials using ramanujan approx

        compute log(prod(xi!)) = (-1) sum(log(x!))
        """
        from distributions.helperFunctions.multinomailApprox import ramanujanApproxNumpy

        loggedVals = ramanujanApproxNumpy(permutation)
        summedVals = loggedVals.sum()
        return -summedVals
    
    def computeStirlingApproxSinglePerm(self, permutation):
        """
        Compute the coefficient using stirlings approximation
        """
        from distributions.helperFunctions.multinomailApprox import stirlingApproxNumpy

        loggedVals = stirlingApproxNumpy(permutation)
        summedVals = loggedVals.sum()
        return -summedVals


    def computePermExpCoef(self, numSamples, maxCount, approxMethod="ramanujan"):
        """
        Compute the product of the factorials of all permutations compute using log and exp at the end for a fixed value of m
        Compute the log prob and sum for each element in the subset of X = {maj(x)= x_bias}
        
        return the exponented value as the logprob is calculated, so retrieve the real prob value

        """        
        from distributions.helperFunctions.multinomailApprox import ramanujanApprox2, stirlingApprox2

        p = self.p
        q = self.q

        if approxMethod == "ramanujan":
            fullCoefficient = ramanujanApprox2(numSamples) - ramanujanApprox2(maxCount)  +  maxCount*math.log(p)  + (numSamples - maxCount)*math.log(q)
        else:
            fullCoefficient = stirlingApprox2(numSamples) - stirlingApprox2(maxCount) + maxCount*math.log(p) + (numSamples - maxCount)*math.log(q)
        
        remainderCoef = 0

        for perm in self.permuter(numSamples-maxCount, maxCount):
            if approxMethod == "ramanujan":
                tempRes = self.computeRamanApproxSinglePerm(np.array(perm))
                remainderCoef += tempRes
                # print("remainder coef", remainderCoef, flush=True)
            else:
                tempRes = self.computeStirlingApproxSinglePerm(np.array(perm))
                remainderCoef += tempRes

            # print("permutation ", perm)
            # print("coefficient ", 1/math.exp(tempRes))
            # break

        return math.exp(fullCoefficient + remainderCoef)
    
    def computePermCoef(self, numSamples, maxCount, approxMethod="ramanujan"):
        """
        Compute the product of the factorials of all permutations compute using log and exp at the end for a fixed value of m

        return the logprob
        """        
        from distributions.helperFunctions.multinomailApprox import ramanujanApprox2, stirlingApprox2

        p = self.p
        q = self.q

        if approxMethod == "ramanujan":
            fullCoefficient = ramanujanApprox2(numSamples) - ramanujanApprox2(maxCount)  +  maxCount*math.log(p)  + (numSamples - maxCount)*math.log(q)
        else:
            fullCoefficient = stirlingApprox2(numSamples) - stirlingApprox2(maxCount) + maxCount*math.log(p) + (numSamples - maxCount)*math.log(q)
        
        remainderCoef = 0

        for perm in self.permuter(numSamples-maxCount, maxCount):
            if approxMethod == "ramanujan":
                tempRes = self.computeRamanApproxSinglePerm(np.array(perm))
                remainderCoef += tempRes
            else:
                tempRes = self.computeStirlingApproxSinglePerm(np.array(perm))
                remainderCoef += tempRes

        return fullCoefficient + remainderCoef


    def probCalculator(self, numSamples, approxMethod="ramanujan"):
        """
        Probability of getting a plurality vote 
        Brute force go through all probabilities for a single number of samples, and the success probability fixed
        """
        from tqdm import tqdm
        numCats = self.numCats
        
        
        if numSamples % 2 == 0:
            majority = numSamples // 2 + 1
        else:
            majority = math.ceil(numSamples / 2)

        
        minPlurality = math.ceil(numSamples // numCats) + 2


        totalProb = 0
        
        for mCount in tqdm(range(minPlurality, numSamples)):
            tempRes = self.computePermExpCoef(numSamples, mCount, approxMethod=approxMethod)
            print("probability of current iteration of m", tempRes, flush=True)
            totalProb += tempRes
            # print("total probability", totalProb, flush=True)
            # break
        return totalProb
    
    def logProbCalculator(self, numSamples, approxMethod="ramanujan"):
        """
        Calculate the approx sum of probabialites using the logexpsum of the log probabilites
        """
        numCats = self.numCats
        if numSamples % numCats == 0:
            minPlurality = math.ceil(numSamples // numCats) + 1
        else:
            minPlurality = math.ceil(numSamples//numCats) + 2

        logProbs = []
        for mCount in tqdm(range(minPlurality, numSamples)):
            tempRes = self.computePermCoef(numSamples, mCount, approxMethod=approxMethod)
            logProbs.append(tempRes)

        totalapproxProb = self.computeLogSumApprox(np.array(logProbs))
        return totalapproxProb
    
    def minSampleLogProbCalculator(self, q, startIndex=None, approxMethod="ramanujan"):
        """
        Compute the probability approximation using approximate logprob and approximate sum of logprob

        """

        if self.p > q:
            return 1
        
        if startIndex is None:
            currentSampleNumber = 3
        else:
            if isinstance(startIndex, int):
                if startIndex > 0:
                    currentSampleNumber = startIndex
                else:
                    raise ValueError("numSamples needs to be positive")
            else:
                raise TypeError("numSamples needs to be an integer")
            
        allSamples= {}
        while True:
            tempProb = self.logProbCalculator(currentSampleNumber, approxMethod=approxMethod)
            print("current Probability currentSampleNumber", tempProb, flush=True)
            allSamples[currentSampleNumber] = tempProb
            if tempProb > q:
                break

            currentSampleNumber += 1

        
        return allSamples , currentSampleNumber

    
    def minSampleCalculator(self, q, startIndex = None, approxMethod="stirling"):
        """
        naive calculator for the minimum number of samples
        Input: a probability q

        Returns the collection of the number of samples, and the lower for required prob value
        """

        if startIndex is None:
            currentSampleNumber = 3
        else:
            if isinstance(startIndex, int):
                if startIndex > 0:
                    currentSampleNumber = startIndex
                else:
                    raise ValueError("numSamples needs to be positive")
            else:
                raise TypeError("numSamples needs to be an integer")

        allSamples= {}
        while True:
            tempProb = self.probCalculator(currentSampleNumber, approxMethod=approxMethod)
            print("current Probability currentSampleNumber", tempProb, flush=True)
            allSamples[currentSampleNumber] = tempProb
            if tempProb > q:
                break

            currentSampleNumber += 1

        
        return allSamples , currentSampleNumber



    def minSampleCalculatorBin(self, q):
        """
        Find the min samples necesary for a plularity level of q
        Search through binary search    
        Inopuyt q
        """

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
                        

        
    def logProbMltinomial(self, numSamples):
        """
        Compute the log probabiliyt of a specific permutation of the pmf
        Input: Permutation
        Number of samples, number of pos result
        """
        from distributions.helperFunctions.multinomailApprox import singlePerm
        p = self.p
        q = self.q
        
        if numSamples % self.numCats == 0:
            lowerBound = numSamples // self.numCats + 1
        else:
            lowerBound = numSamples // self.numCats + 2

        upperBound = numSamples // 2 + 1
      
        totalProb = 0
        for m in tqdm(range(lowerBound, numSamples)):
            if m > upperBound:
                # accept everything
                for perm in self.permuter(numSamples - m):
                    logProb = singlePerm(x=perm, n=numSamples-m, m=m, p =p, q=q)
                    print("log prob: ", logProb, flush=True)
                    totalProb += math.exp(logProb)
            else:
                for perm in self.permuter(numSamples - m):
                    counts = Counter(perm)
                    highCount = counts.most_common(1)
                    if highCount[0][1] >= m:
                        continue
                    else:
                        logProb = singlePerm(x=perm, n=numSamples-m, m=m, p =p, q=q)
                        print("logprob ", logProb, flush=True)
                        totalProb += math.exp(logProb)
        
        return totalProb
    
    def computeLogSumApprox(self, logProbs):
        """
        Compute the approx logsum probability using the approximation 

        sum(pi) = sum(exp(log(pi))) approx exp(m + log(sum(exp(li-m)))
        """

        m = max(logProbs)
        interExp = math.log(sum(np.exp(logProbs - m)))
        return math.exp(m + interExp)


    def computeSanavApprox(self, perm, numSamples, maxCount):
        p = self.p
        q = self.q
        pHat = perm/numSamples
        pBias = maxCount/numSamples

        tempQ = (pHat - q)**2 / q
        tempP = (pBias - p)**2 / p 

        return -numSamples/2 *(tempQ + tempP)



    def findLowerBoundApprox(self, threshold):
        """
        Function to find the lowest value for the 
        
        """
        lowerBound = 1
        upperBound = 10000

        upperBoundFound = False
        estimateResults = {} 
        
        while not upperBoundFound:
            tempProb = self.logProbMltinomial(upperBound)
            estimateResults[upperBound] = tempProb
            if tempProb > threshold:
                upperBoundFound = True
            
            else:
                lowerBound = upperBound
                upperBound = upperBound * 2

        
        midPoint = math.ceil((lowerBound + upperBound) / 2)

        while lowerBound != upperBound:
            tempProb = self.logProbMltinomial(midPoint)
            estimateResults[midPoint] = tempProb
            if tempProb >= threshold:
                lowerBound = midPoint
            else:
                upperBound = midPoint

        return estimateResults, midPoint


        
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
    
