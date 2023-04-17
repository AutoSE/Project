import math
import re
import math
import copy
import json
from pathlib import Path
import cli as c
from sym import Sym
Seed = 937162211

def settings(s):
    return dict(re.findall("\n[\s]+[-][\S]+[\s]+[-][-]([\S]+)[^\n]+= ([\S]+)",s))

#Numerics
def rint(lo, hi, mseed=None):
    return math.floor(0.5 + rand(lo, hi, mseed))

def rand(lo, hi, mseed=None):
    lo= lo or 0
    hi= hi or 1
    global Seed
    Seed = 1 if mseed else (16807 * Seed) % 2147483647
    return lo + (hi-lo) * Seed / 2147483647

def rnd(n, nPlaces=3    ):
    mult = 10**nPlaces 
    return math.floor(n * mult + 0.5) / mult

def cosine(a, b, c):
    x1 = (a**2 + c**2 - b**2) / (2**c)
    x2 = max(0, min(1, x1))
    y  = (a**2 - x2**2)**0.5
    return x2, y


#Strings
def fmt(sControl):
    return str(sControl)

def oo(t):
    d = t.__dict__
    d['a'] = t.__class__.__name__
    d['id'] = id(t)
    #d = dict(sorted(d.items()))
    print(dict(sorted(d.items())))

def coerce(s):
    if s=='true' or s=='True':
        return True
    elif s=='false' or s=='False':
        return False
    elif s.isdigit():
        return int(s)
    elif '.' in s:
        if s.replace('.','').isdigit():
            return float(s)
        else:
            return s
    else:
        return s

def csv(sFilename, fun):
    sFilename=Path(sFilename)
    t=[]
    f=open(sFilename.absolute(),'r')
    lines=f.readlines()
    for line in lines:
        t=[]
        for s in re.findall("([^,]+)", line):
            t.append(coerce(s))
        fun(t)


#Lists
def map(t, fun):
    u={}
    for v in t:
        k=fun(v)
        if k:
            u[k]=v
        else:
            u[1+len(u)]=v
    return u

def kap(t, fun):
    u={}
    for k,v in enumerate(t):
        v,k=fun(k,v)
        if k:
            u[k]=v
        else:
            u[1+len(u)]=v
    return u

def kapd(t, fun):
    u = {}
    for k,v in t.items():
        v, k = fun(k,v)
        u[k or len(u)] = v
    return u

def any(t):
    return t[rint(0,len(t)-1)]

def many(t,n):
   u=[]
   for i in range(1,n+1):
    u.append(any(t))
   return u

def show(node, what, cols, nPlaces, lvl =0):
    if node:
        print('| ' * lvl + str(len(node['data'].rows)) + '  ', end = '')
        if not node.get('left') or lvl==0:
            print(node['data'].stats("mid",node['data'].cols.y,nPlaces))
        else:
            print('')
        show(node.get('left'), what,cols, nPlaces, lvl+1)
        show(node.get('right'), what,cols,nPlaces, lvl+1)


def merge(col1,col2):
  new = copy.deepcopy(col1)
  if isinstance(col1, Sym):
      for n in col2.has:
        new.add(n)
  else:
    for n in col2.has:
        new.add(new,n)
    new.lo = min(col1.lo, col2.lo)
    new.hi = max(col1.hi, col2.hi) 
  return new

def RANGE(at,txt,lo,hi=None):
    return {'at':at,'txt':txt,'lo':lo,'hi':lo or hi or lo,'y':Sym()}

def extend(range,n,s):
    range['lo'] = min(n, range['lo'])
    range['hi'] = max(n, range['hi'])
    range['y'].add(s)

def itself(x):
    return x

def value(has,nB = None, nR = None, sGoal = None):
    sGoal,nB,nR = sGoal or True, nB or 1, nR or 1
    b,r = 0,0
    for x,n in has.items():

        if x==sGoal:
            b = b + n
        else:
            r = r + n
    b,r = b/(nB+1/float("inf")), r/(nR+1/float("inf"))
    return b**2/(b+r)


def merge2(col1,col2):
  new = merge(col1,col2)
  if new.div() <= (col1.div()*col1.n + col2.div()*col2.n)/new.n:
    return new

def cliffsDelta(ns1,ns2):
    if len(ns1) > 256:
        ns1 = many(ns1,256)
    if len(ns2) > 256:
        ns2 = many(ns2,256)
    if len(ns1) > 10*len(ns2):
        ns1 = many(ns1,10*len(ns2))
    if len(ns2) > 10*len(ns1):
        ns2 = many(ns2,10*len(ns1))
    n,gt,lt = 0,0,0
    for x in ns1:
        for y in ns2:
            n = n + 1
            if x > y:
                gt = gt + 1
            if x < y:
                lt = lt + 1
    return abs(lt - gt)/n > float(c.the['cliffs'])

def showTree(tree, what, cols, nPlaces, lvl = 0):
  if tree:
    print('|.. ' * lvl + '[' + str(len(tree['data'].rows)) + ']' + '  ', end = '')
    if not tree.get('left') or lvl==0:
        print(tree['data'].stats("mid",tree['data'].cols.y,nPlaces))
    else:
        print('')
    showTree(tree.get('left'), what,cols, nPlaces, lvl+1)
    showTree(tree.get('right'), what,cols,nPlaces, lvl+1)

def bins(cols,rowss):
    out = []
    for col in cols:
        ranges = {}
        for y,rows in rowss.items():
            for row in rows:
                x = row.cells[col.at]
                if x != "?":
                    k = int(bin(col,x))
                    if not k in ranges:
                        ranges[k] = RANGE(col.at,col.txt,x)
                    extend(ranges[k], x, y)
        ranges = list(dict(sorted(ranges.items())).values())
        r = ranges if isinstance(col, Sym) else mergeAny(ranges)
        out.append(r)
    return out

def bin(col,x):
    if x=="?" or isinstance(col, Sym):
        return x
    tmp = (col.hi - col.lo)/(c.the['bins'] - 1)
    return  1 if col.hi == col.lo else math.floor(x/tmp + .5)*tmp

def mergeAny(ranges0):
    def noGaps(t):
        for j in range(1,len(t)):
            t[j]['lo'] = t[j-1]['hi']
        t[0]['lo']  = float("-inf")
        t[len(t)-1]['hi'] =  float("inf")
        return t
    ranges1,j = [],0
    while j <= len(ranges0)-1:
        left = ranges0[j]
        right = None if j == len(ranges0)-1 else ranges0[j+1]
        if right:
            y = merge2(left['y'], right['y'])
            if y:
                j = j+1
                left['hi'], left['y'] = right['hi'], y
        ranges1.append(left)
        j = j+1
    return noGaps(ranges0) if len(ranges0)==len(ranges1) else mergeAny(ranges1)

def prune(rule, maxSize):
    n=0
    for txt, ranges in rule.items():
        n=n+1
        if len(ranges)== maxSize[txt]:
            n=n-1
            rule['txt']=None
    if n>0:
        return rule