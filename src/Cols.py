import math
from sym import Sym
from num import Num

class Cols:
    def __init__(self,t):
        self.names, self.all, self.x, self.y, self.klass = t, [], [], [], None
        for n,s in enumerate(t):
            col = Num(n,s) if s[0].isupper() else Sym(n,s)
            self.all.append(col)

            if s[-1] != 'X':
                if '!' in s:
                    self.klass = col
                self.y.append(col) if '!' in s or '+' in s or '-' in s else self.x.append(col)
                
    def add(self,row):
        for t in [self.x, self.y]:
            for col in t:
                col.add(row.cells[col.at])
