import random
import Utils as u
import Cols as c
import cli as g
import Row as r
import num as n
import math
import numpy as np
from operator import itemgetter
from functools import cmp_to_key
from sklearn.cluster import AgglomerativeClustering,SpectralClustering


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
        def fun(_,col):
            if what=='mid':
                val=col.mid()
            else:
                val=col.div()
            return col.rnd(val, nPlaces),col.txt
        return u.kap(cols or self.cols.y, fun)

    def better1(self, row1, row2):
        s1,s2,ys=0,0,self.cols.y
        for col in ys:
            x=col.norm(row1.cells[col.at])
            y=col.norm(row2.cells[col.at])
            s1=s1-math.exp(col.w*(x-y)/len(ys))
            s2=s2-math.exp(col.w*(y-x)/len(ys))
        return s1/len(ys) < s2/len(ys)

    def better(self, rows1, rows2, s1=0, s2=0, ys=None, x=0, y=0):
        if isinstance(rows1, r.Row):
            rows1 = [rows1]
            rows2 = [rows2]
        if not ys:
            ys = self.cols.y
        for col in ys:
            for row1, row2 in zip(rows1, rows2):
                x = col.norm(row1.cells[col.at])
                y = col.norm(row2.cells[col.at])
                s1 = s1 - math.exp(col.w * (x - y) / len(ys))
                s2 = s2 - math.exp(col.w * (y - x) / len(ys))
        return s1 / len(ys) < s2 / len(ys)
    


    def dist(self, row1, row2, cols=None):
        n,d = 0,0
        for col in cols or self.cols.x:
            n = n + 1
            d = d + col.dist(row1.cells[col.at], row2.cells[col.at])**int(g.the["p"])
        return (d/n)**(1/int(g.the["p"]))

    def around(self, row1, rows=None, cols=None):
        def function(row2):
            return {'row' : row2, 'dist' : self.dist(row1,row2,cols)} 
        return sorted(list(map(function, rows or self.rows)), key= itemgetter('dist'))

    def half(self, rows=None, cols=None, above=None):
        def gp(row1, row2):
            return self.dist(row1, row2, cols)
        def project(row):
            return {'row': row, 'dist': u.cosine(gp(row, A), gp(row,B), c)}
        rows=rows or self.rows
        some=u.many(rows, int(g.the['Halves']))
        A = above if above else u.any(some)
        def function(r):
            return {'row' : r, 'dist' : gp(r, A)}
        tmp = sorted(list(map(function, some)), key=itemgetter('dist'))
        far = tmp[int(float(g.the['Far']) * len(rows))]
        B=far['row']
        c=far['dist']
        left,right=[],[]
        for n, tmp in enumerate(sorted(list(map(project, rows)), key=itemgetter('dist'))):
            if n<len(rows)//2:
                left.append(tmp['row'])
            else:
                right.append(tmp['row'])
        evals=1 if g.the['Reuse']=='true' and above else 2
        return left, right, A,B,c,evals

    def agglomerative_clustering(self, rows=None):
        left = []
        right = []

        if not rows:
            rows = self.rows
        row_set = np.array([r.cells for r in rows])
        agg_clust = AgglomerativeClustering(n_clusters=2,metric='euclidean',linkage='ward')
        agg_clust.fit(row_set)

        for key, value in enumerate(agg_clust.labels_):
            if value == 0:
                left.append(rows[key])
            else:
                right.append(rows[key])
        return left, right, random.choices(left, k=10), random.choices(right, k=10), 1
    

    def spectral_clustering(self, rows=None):
        left = []
        right = []

        if not rows:
            rows = self.rows
        row_set = np.array([r.cells for r in rows])

        sc = SpectralClustering(n_clusters=2,affinity='rbf').fit(row_set)

        for key, value in enumerate(sc.labels_):
            if value == 0:
                left.append(rows[key])
            else:
                right.append(rows[key])
        return left, right, random.choices(left, k=10), random.choices(right, k=10), 1
    


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

    def sway1(self, rows=None, min=None, cols=None, above=None):
        data = self
        def worker(rows, worse, evals0, above = None):
            if len(rows) <= len(data.rows)**float(g.the['min']): 
                return rows, u.many(worse, int(g.the['rest'])*len(rows)),evals0
            else:
                l,r,A,B,evals = self.agglomerative_clustering(rows)
                if self.better(B,A):
                    l,r,A,B = r,l,B,A
                for row in r:
                    worse.append(row)
                return worker(l,worse,evals+evals0,A)
        best, rest, evals = worker(data.rows,[],0)
        return self.clone(best), self.clone(rest), evals
    


    def sway2(self, rows=None, min=None, cols=None, above=None):
        data = self
        def worker(rows, worse, evals0, above = None):
            if len(rows) <= len(data.rows)**float(g.the['min']): 
                return rows, u.many(worse, int(g.the['rest'])*len(rows)),evals0
            else:
                l,r,A,B,evals = self.spectral_clustering(rows)
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
            while j<=len(t0):
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
            if ranges is not None:
                return list(map(pretty,merge(sorted(ranges,key=itemgetter('lo'))))),attr
        return u.kapd(rule,merges)


    def firstN(self, sortedRanges, scoreFun):
        print()
        def function(r):
            print(r['range']['txt'],r['range']['lo'],r['range']['hi'],u.rnd(r['val']),r['range']['y'].has)
        _ = list(map(function, sortedRanges))
        print()
        first = sortedRanges[0]['val']
        def useful(range):
            if range['val']>.05 and range['val']> first/10:
                return range
        sortedRanges = [x for x in sortedRanges if useful(x)]
        most,out = -1, -1
        for n in range(1,len(sortedRanges)+1):
            slice = sortedRanges[0:n]
            slice_range = [x['range'] for x in slice]
            tmp,rule = scoreFun(slice_range)
            if tmp and tmp > most:
                out,most = rule,tmp
        return out,most

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
            maxSizes[ranges[0]['txt']] = len(ranges)
            print("")
            for range in ranges:
                print(range['txt'], range['lo'], range['hi'])
                tmp.append({'range':range, 'max':len(ranges),'val': v(range['y'].has)})
        rule,most=self.firstN(sorted(tmp, key=itemgetter('val')),score)
        return rule,most

    def selects(self, rule, rows):
        def disjunction(ranges, row):
            if ranges is None:
                return False
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