import math

class Sym:
    def __init__(self, at=None, txt=None):
        self.n = 0
        self.most = 0
        self.mode = None
        self.has = {}
        self.at , self.txt = at if at else 0, txt if txt else ""
        
    def add(self,x):
        if x != '?':
            self.n = self.n + 1
            self.has[x] = 1 + (self.has[x] if x in self.has.keys() else 0)
            if self.has[x] > self.most:
                self.most,self.mode = self.has[x],x

    def mid(self):
        return self.mode

    def div(self):
        def fun(p):
            return p*math.log(p,2)
        e=0
        for k,v in self.has.items():
            e = e + fun(v/self.n)
        return -1*e

    def rnd(self,x,n):
        return x

    def dist(self, s1, s2):
        if s1 == "?" and s2 == "?":
            return 1
        elif s1 == s2:
            return 0
        else: 
            return 1


