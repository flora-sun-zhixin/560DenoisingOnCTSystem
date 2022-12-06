import numpy as np
from utils.util import *

class Circular:
    def __init__(self, c, sigma, amplitude):
        self.c = c
        self.sigma = sigma
        self.amplitude = amplitude
    
    def getCircular(self, location):
        return self.amplitude * rect(location) * circ(location, self.c, self.sigma)

class Lumps:
    def __init__(self, N, sigma=1/4, amplitude=1):
        self.N = N
        self.c = Lumps.getRandomC(N)
        self.sigma = sigma
        self.amplitude = amplitude
    
    @staticmethod
    def getALump(c, sigma, amplitude, location):
        return amplitude / (2 * np.pi * sigma**2) * np.exp(- np.linalg.norm(location - c, ord=2, axis=-1) / (2 * sigma**2))

    def getLumps(self, location):
        holder = np.zeros(location.shape[:2])
        for i in range(self.N):
            holder += Lumps.getALump(self.c[i], self.sigma, self.amplitude, location)
        return rect(location) * holder

    @staticmethod
    def getRandomC(N):
        return np.random.uniform(0, 1, (N, 2)) - 1/2

class SimuObject:
    def __init__(self, N, b_sigma=1/4, b_amplitude=1,
                 s_c=np.array([0, 0]), s_sigma=1/4, s_amplitude=3):
        self.signal = Circular(s_c, s_sigma, s_amplitude)
        self.background = Lumps(N, b_sigma, b_amplitude)

    def getSignal(self, location):
        return self.signal.getCircular(location)

    def getObject(self, location, kind):
        if kind == "SignalAbsent":
            return self.getSignalAbsentObject(location)
        elif kind == "SignalPresent":
            return self.getSignalPresentObject(location)
        else:
            raise f"{kind} is not implemented."

    def getSignalAbsentObject(self, location):
        return self.background.getLumps(location)

    def getSignalPresentObject(self, location):
        return self.background.getLumps(location) + self.signal.getCircular(location)
