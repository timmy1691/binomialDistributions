import numpy as np
import math

def binomial(p, n):
    """
    Binomail distribution to find the number of samples with success probability and result probability q
    """

    minK = n//2 + 1
    totalProbability = 0
    for k in range(minK, n+1):
        totalProbability += math.comb(n, k) * (p ** k) * ((1 - p) ** (n - k))
    
    return totalProbability


def naiveMinSamples(p, q):
    """
    Naive method of finding 
    """

    if p <= 0.5:
        raise ValueError("Probability of success must be greater than 0.5")
    
    if q <= p :
        return 1

    currentProb = 0
    currentN = 1
    while currentProb < q:
        resProb = binomial(p, currentN)
        if resProb >= q:
            return currentN
        else:
            currentN += 1




def minSamples(p, q):
    """
    Is there a more clever way than naively searching?
    """


    