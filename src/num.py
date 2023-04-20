import re
import Utils as u
import cli as g
import math
class Num:
    def __init__(self, at=0, txt="", t=None):
        self.at, self.txt = at, txt 
        self.n, self.mu, self.m2 = 0,0,0
        self.mu=0
        self.n=0
        self.lo, self.hi = float("inf"), float("-inf")
        self.w = -1 if "-" in self.txt else 1 
        self.has = {}
        self.sd = 0
        if t:
            for n in t:
                self.add(n)
    def add(self,n):
        if n != '?':
            n=float(n)
            self.n = self.n + 1
            if self.n <= int(g.the['Max']):
                self.has[n]= n
            d = n - self.mu
            self.mu = self.mu + d/self.n
            self.m2 = self.m2 + d*(n-self.mu)
            self.lo = min(n, self.lo)
            self.hi = max(n, self.hi)
            self.sd = 0 if self.n<2 else (self.m2/(self.n - 1))**.5
    def mid(self):
        return self.mu

    def div(self):
        if (self.m2 < 0 or self.n < 2):
            return 0
        else:
            return math.sqrt(self.m2 / (self.n - 1))

    def rnd(self,x,n):
        return x if x=='?' else u.rnd(x,n)

    def norm(self, n):
        return n if n == "?" else (n-self.lo)/(self.hi - self.lo + 10**-32)

    def dist(self, n1, n2):
        if n1 == "?" and n2 == "?":
            return 1
        n1, n2 = self.norm(n1), self.norm(n2)
        if n1 == "?":
            n1 = 1 if n2 < 0.5 else 0
        if n2 == "?":
            n2 = 1 if n1 < 0.5 else 0
        return abs(n1 - n2)
    
    def vals(self):
        return list(dict(sorted(self.has.items(), key=lambda x: x[1])).values())