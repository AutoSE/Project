import math
from sym import Sym
from num import Num

class Cols:
    def __init__(self,t):
        self.names, self.all, self.x, self.y, self.klass = t, [], [], [], None
        for col_name in t:
            if col_name[0].isupper():
                col = Num(t.index(col_name), col_name)
            else:
                col = Sym(t.index(col_name), col_name)
            self.all.append(col)
            if not col_name[-1] == "X":
                if "-" in col_name or "+" in col_name or "!" in col_name:
                    self.y.append(col)
                else:
                    self.x.append(col)
                if "!" in col_name:
                    self.klass=col
                
    def add(self,row):
        for t in [self.x, self.y]:
            for col in t:
                col.add(row.cells[col.at])
