import math
from scipy.stats import norm
import pandas as pd 
from datetime import date
import numpy as np 

class option:
    def __init__(self,right, s, k, eval_date, exp_date, price = None, rf = 0.01, sigma = 0.3, div = 0):
        self.k = float(k)
        self.s = float(s)
        self.rf = float(rf)
        self.sigma = float(sigma)
        self.eval_date = eval_date
        self.exp_date = exp_date
        self.t = self.calculate_t()
        if self.t == 0: self.t = 0.000001
        self.price = price
        self.right = right ## 'C' or 'P'
        self.div = div
    def calculate_t(self):
        if isinstance(self.eval_date, basestring):
            if '/' in self.eval_date:
                (day, month, year) = self.eval_date.split('/')
            else:
                (day, month, year) = self.eval_date[6:8], self.eval_date[4:6],self.eval_date[0:4]
            d0 = date(int(year), int(month), int(day))
        elif type(self.eval_date) == float or type(self.eval_date) == long or type(self.eval_date) == np.float64:
            (day, month, year) = (str(self.eval_date)[6:8], str(self.eval_date)[4:6], str(self.eval_date)[0:4])
            d0 = date(int(year), int(month), int(day))
        else:
            d0 = self.eval_date
        if isinstance(self.exp_date, basestring):
            if '/' in self.exp_date:
                (day, month, year) = self.exp_date.split('/')
            else:
                (day, month, year) = self.exp_date[6:8], self.exp_date[4:6],self.exp_date[0:4]
            d1 = date(int(year), int(month), int(day))
        elif type(self.exp_date) == float or type(self.exp_date) == long or type(self.exp_date) == np.float64:
            (day, month, year) = (str(self.exp_date)[6:8], str(self.exp_date)[4:6], str(self.exp_date)[0:4])
            d1 = date(int(year), int(month), int(day))
        else:
            d1 = self.exp_date

        return (d1-d0).day / 365

    def get_price_delta(self):
        d1 = (math.log(self.s/self.k) + (self.rf + self.div + math.pow(self.sigma,2)/2 * self.t)/ (self.sigma * math.sqrt(self.t)))
        d2 = d1 - self.sigma * math.sqrt(self.t)
        if self.right == 'C':
            self.calc_price = (norm.cdf(d1) * self.s * math.exp(-self.div * self.t) - norm.cdf(d2) * self.k * math.exp(-self.rf * self.t))
            self.delta = norm.cdf(d1)
        elif self.right == 'P':
            self.calc_price = (-norm.cdf(-d1) * self.s * math.exp(-self.div * self.t) + norm.cdf(-d2) * self.k * math.exp(-self.rf * self.t))
            self.delta = -norm.cdf(-d1)
    
    def get_call(self):
        d1 = (math.log(self.s/self.k) + (self.rf + self.div + math.pow(self.sigma,2)/2 * self.t)/ (self.sigma * math.sqrt(self.t)))
        d2 = d1 - self.sigma * math.sqrt(self.t)
        self.calc_price = (norm.cdf(d1) * self.s * math.exp(-self.div * self.t) - norm.cdf(d2) * self.k * math.exp(-self.rf * self.t))
        self.call_delta = norm.cdf(d1)
    
    def get_put(self):
          def get_price_delta(self):
        d1 = (math.log(self.s/self.k) + (self.rf + self.div + math.pow(self.sigma,2)/2 * self.t)/ (self.sigma * math.sqrt(self.t)))
        d2 = d1 - self.sigma * math.sqrt(self.t)
        self.calc_price = (-norm.cdf(-d1) * self.s * math.exp(-self.div * self.t) + norm.cdf(-d2) * self.k * math.exp(-self.rf * self.t))
        self.put_delta = -norm.cdf(-d1)



    