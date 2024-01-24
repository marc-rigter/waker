import random
import math
import numpy as np
from enum import Enum

class Interp(Enum):
  COSINE = 0
  CUBIC = 1
  LINEAR = 2
  
def generate_step_profile(num_points, size, width=0.4):
    """ Generate height profile for step terrain. """
    step_height = 0.5
    step_width = 0.4
    
    increment = size / num_points
    x = 0.0
    y = 0.5
    sign = 1
    step = 0
    ys = []
    for _ in range(num_points):
      x += increment
      if x / step_width > step:
        step += 1
        if y < 0.0 or y > 1.0:
          sign = sign * -1
        y += step_height * sign
      ys.append(y)
    ys = np.clip(np.array(ys), 0.0, 1.0)
    return np.expand_dims(ys, 1)

class PerlinNoise():
    def __init__(self, amplitude=1, frequency=1, 
            octaves=1, interp="COSINE", use_fade=False):
        assert interp == "COSINE" or interp == "CUBIC" or interp == "LINEAR"
        self.amplitude = amplitude
        self.frequency = frequency
        self.octaves = octaves
        self.interp = interp
        self.use_fade = use_fade
        self.mem_x = dict()

    def __noise(self, x):
        # made for improve performance
        if x not in self.mem_x:
            self.mem_x[x] = random.Random(x).uniform(-1, 1)
        return self.mem_x[x]

    def __interpolated_noise(self, x):
        prev_x = int(x) # previous integer
        next_x = prev_x + 1 # next integer
        frac_x = x - prev_x # fractional of x

        if self.use_fade:
            frac_x = self.__fade(frac_x)

        # intepolate x
        if self.interp == "LINEAR":
            res = self.__linear_interp(
                self.__noise(prev_x), 
                self.__noise(next_x),
                frac_x)
        elif self.interp == "COSINE":
            res = self.__cosine_interp(
                self.__noise(prev_x), 
                self.__noise(next_x),
                frac_x)
        else:
            res = self.__cubic_interp(
                self.__noise(prev_x - 1), 
                self.__noise(prev_x), 
                self.__noise(next_x),
                self.__noise(next_x + 1),
                frac_x)

        return res

    def get(self, x):
        frequency = self.frequency
        amplitude = self.amplitude
        result = 0
        for _ in range(self.octaves):
            result += self.__interpolated_noise(x * frequency) * amplitude
            frequency *= 2
            amplitude /= 2

        return result

    def __linear_interp(self, a, b, x):
        return a + x * (b - a)

    def __cosine_interp(self, a, b, x):
        x2 = (1 - math.cos(x * math.pi)) / 2
        return a * (1 - x2) + b * x2

    def __cubic_interp(self, v0, v1, v2, v3, x):
        p = (v3 - v2) - (v0 - v1)
        q = (v0 - v1) - p
        r = v2 - v0
        s = v1
        return p * x**3 + q * x**2 + r * x + s

    def __fade(self, x):
        # useful only for linear interpolation
        return (6 * x**5) - (15 * x**4) + (10 * x**3)