import Utils as u
import Cols as c
import cli as g
import Row as r
import num as n
import math
from operator import itemgetter
class Data:
    def __init__(self, src):
        self.rows=[]
        self.cols=None
        if isinstance(src, str):
            u.csv(src, self.add)
        else:
            for row in src:
                self.add(row)

    def add(self,t):
        if self.cols:
            t=r.Row(t) if type(t)==list else t
            self.rows.append(t)
            self.cols.add(t)
        else:
            self.cols=c.Cols(t)

    def clone(self, init):
        data = Data([self.cols.names])
        _ = list(map(data.add, init))
        return data

    def stats(self, what, cols, nPlaces):
        def fun(k,col):
            if what=='mid':
                val=col.mid()
            else:
                val=col.div()
            return col.rnd(val, nPlaces),col.txt
        if cols:
            return u.kap(cols, fun)
        else:
            return u.kap(self.cols.y, fun)

    def better(self, row1, row2):
        s1,s2,ys=0,0,self.cols.y
        for col in ys:
            x=col.norm(row1.cells[col.at])
            y=col.norm(row2.cells[col.at])
            s1=s1-math.exp(col.w*(x-y)/len(ys))
            s2=s2-math.exp(col.w*(y-x)/len(ys))
            return s1/len(ys) < s2/len(ys)

    def dist(self, row1, row2, cols=None):
        n,d = 0,0
        for col in (cols or self.cols.x):
            n = n + 1
            d = d + col.dist(row1.cells[col.at], row2.cells[col.at])**g.the["p"]
        return (d/n)**(1/g.the["p"])

    def around(self, row1, rows=None, cols=None):
        def function(row2):
            return {'row' : row2, 'dist' : self.dist(row1,row2,cols)} 
        return sorted(list(map(function, rows or self.rows)), key= lambda k: k['dist'])

    def half(self, rows=None, cols=None, above=None):
        def dist(row1, row2):
            return self.dist(row1, row2, cols)
        rows=rows or self.rows
        some=u.many(rows, g.the['Sample'])
        A=above or u.any(some)
        B=self.around(A, some)[int((float(g.the['Far'])*len(rows)))]['row']
        c=dist(A,B)
        left,right=[],[]
        
        def project(row):
            return {'row': row, 'dist': u.cosine(dist(row, A), dist(row,B), c)}
        for n, tmp in enumerate(sorted(list(map(project, rows)), key=itemgetter('dist'))):
            if n<=len(rows)//2:
                left.append(tmp['row'])
                mid=tmp['row']
            else:
                right.append(tmp['row'])
        return left, right, A,B,mid,c

    def cluster(self, rows = None, min=None, cols=None, above=None):
        rows = rows or self.rows
        min = min or (len(rows)**float(g.the['min']))
        cols = cols or self.cols.x
        node = {'data' : self.clone(rows)}
        if len(rows) >= 2*min:
            left, right, node['A'], node['B'], node['mid'], _ = self.half(rows,cols,above)
            node['left'] = self.cluster(left, min, cols, node['A'])
            node['right'] = self.cluster(right, min, cols, node['B'])
        return node

    def sway(self, rows=None, min=None, cols=None, above=None):
        data = self
        def worker(rows, worse, evals0, above = None):
            if len(rows) <= len(data.rows)**float(g.the['min']): 
                return rows, u.many(worse, int(g.the['rest'])*len(rows)),evals0
            else:
                l,r,A,B,_,evals = self.half(rows, None, above)
                if self.better(B,A):
                    l,r,A,B = r,l,B,A
                for row in r:
                    worse.append(row)
                return worker(l,worse,evals+evals0,A)
        best, rest, evals = worker(data.rows,[],0)
        return self.clone(best), self.clone(rest), evals
    
    def tree(self, rows = None , min = None, cols = None, above = None):
        rows = rows or self.rows
        min  = min or len(rows)**float(g.the['min'])
        cols = cols or self.cols.x
        node = { 'data' : self.clone(rows) }
        if len(rows) >= 2*min:
            left, right, node['A'], node['B'], node['mid'], _ = self.half(rows,cols,above)
            node['left']  = self.tree(left,  min, cols, node['A'])
            node['right'] = self.tree(right, min, cols, node['B'])
        return node
    
    def showRule(self, rule):
        def pretty(range):
            return range['lo'] if range['lo']==range['hi'] else [range['lo'], range['hi']]
        def merge(t0):
            t,j=[],1
            while j<len(t0):
                left=t0[j-1]
                if j<len(t0):
                    right=t0[j]
                else:
                    right=None
                if right and left['hi'] == right['lo']:
                    left['hi'] = right['hi']
                    j=j+1
                t.append({'lo':left['lo'], 'hi':left['hi']})
                j=j+1
            return t if len(t0)==len(t) else merge(t) 
        def merges(attr, ranges):
            return list(map(pretty,merge(sorted(ranges,key=itemgetter('lo'))))),attr
        return u.kapd(rule,merges)


    def firstN(self, sorted_ranges, scoreFun):
        print()
        for r in sorted_ranges:
            print(r['range']['txt'], r['range']['lo'], r['range']['hi'], u.rnd(r['val']), dict(r['range']['y'].has))
        first = sorted_ranges[0]['val']
        def useful(range):
            if range['val'] > 0.05 and range['val'] > first / 10:
                return range
        sorted_ranges = [s for s in sorted_ranges if useful(s)]
        most: int = -1
        out: int = -1
        for n in range(len(sorted_ranges)):
            tmp, rule = scoreFun([r['range'] for r in sorted_ranges[:n+1]])
            if tmp is not None and tmp > most:
                out, most = rule, tmp
        return out, most

    def betters(self,n):
        tmp=sorted(self.rows, key=lambda row: self.better(row, self.rows[self.rows.index(row)-1]))
        return  n and tmp[0:n], tmp[n+1:]  or tmp

    def RULE(self, ranges, maxSize):
        t={}
        for range in ranges:
            t[range['txt']]= t.get(range['txt']) or []
            t[range['txt']].append({'lo' : range['lo'],'hi' : range['hi'],'at':range['at']})
        return u.prune(t, maxSize)

    def xpln(self,best,rest):
        tmp,maxSizes = [],{}
        def v(has):
            return u.value(has, len(best.rows), len(rest.rows), "best")
        def score(ranges):
            rule = self.RULE(ranges,maxSizes)
            if rule:
                print(self.showRule(rule))
                bestr= self.selects(rule, best.rows)
                restr= self.selects(rule, rest.rows)
                if len(bestr) + len(restr) > 0: 
                    return v({'best': len(bestr), 'rest':len(restr)}),rule
        for ranges in u.bins(self.cols.x,{'best':best.rows, 'rest':rest.rows}):
            maxSizes[ranges[1]['txt']] = len(ranges)
            print("")
            for range in ranges:
                print(range['txt'], range['lo'], range['hi'])
                tmp.append({'range':range, 'max':len(ranges),'val': v(range['y'].has)})
        rule,most=self.firstN(sorted(tmp, key=itemgetter('val')),score)
        return rule,most

    def selects(self, rule, rows):
        def disjunction(ranges, row):
            for range in ranges:
                lo, hi, at = range['lo'], range['hi'], range['at']
                x = row.cells[at]
                if x == "?":
                    return True
                if lo == hi and lo == x:
                    return True
                if lo <= x and x < hi:
                    return True
            return False
        def conjunction(row):
            for ranges in rule.values():
                if not disjunction(ranges, row):
                    return False
            return True
        def function(r):
            if conjunction(r):
                return r
        return list(map(function, rows))