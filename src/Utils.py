import math
import re
import math
import copy
import json
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import cli as w
import num as q
from sym import Sym
Seed = 937162211
from scipy.stats import kruskal, mannwhitneyu
import random
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

def rnd(n, nPlaces=3):
    mult = 10**nPlaces 
    return math.floor(n * mult + 0.5) / mult

def cosine(a, b, c):
    den=1 if c==0 else 2*c
    x1 = (a**2 + c**2 - b**2) / den

    return x1


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
    elif '.' in s and s.replace('.','').isdigit():
        return float(s)
    else:
        return s

def csv(sFilename, fun):
    sFilename=Path(sFilename)
    if sFilename.exists() and sFilename.suffix == '.csv':
        t = []
        with open(sFilename.absolute(), 'r', encoding='utf-8') as file:
            for _, line in enumerate(file):
                row = list(map(coerce, line.strip().split(',')))
                t.append(row)
                fun(row)
    else:
        print("File path does not exist OR File not csv, given path: ", sFilename.absolute())
        return

#Lists

def kap(t, fun):
    u = {}
    for v in t:
        k = t.index(v)
        v, k = fun(k,v) 
        u[k or len(u)] = v
    return u

def kapd(t, fun):
    u = {}
    for k,v in t.items():
        if k is not None and v is not None:
            v, k = fun(k,v)
            u[k or len(u)] = v
    return u

def any(t):
    return t[rint(0,len(t)-1)]

def many(t,n):
   u=[]
   for _ in range(1,n+1):
    u.append(any(t))
   return u

def show(node, what, cols, nPlaces, lvl =0):
    if node:
        print('|..' * lvl, end = '')
        if not node.get('left'):
            print(node['data'].rows[-1].cells[-1])
        else:
            print(int(rnd(100*node['c'], 0)))
        show(node.get('left'), what,cols, nPlaces, lvl+1)
        show(node.get('right'), what,cols,nPlaces, lvl+1)

def deepcopy(t):
    return copy.deepcopy(t)

def merge(col1,col2):
  new = deepcopy(col1)
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
    return abs(lt - gt)/n > float(w.the['cliffs'])

def showTree(tree, what, cols, nPlaces, lvl = 0):
  if tree:
    print('|.. ' * lvl + '[' + str(len(tree['data'].rows)) + ']' + '  ', end = '')
    if not tree.get('left') or lvl==0:
        print(tree['data'].stats("mid",tree['data'].cols.y,nPlaces))
    else:
        print('')
    showTree(tree.get('left'), what,cols, nPlaces, lvl+1)
    showTree(tree.get('right'), what,cols,nPlaces, lvl+1)

def preprocess_data(file, data):
    df = pd.read_csv(file)
    df = impute_missing_values(df)
    label_encoding(df)
    file = file.replace('.csv', '_preprocessed.csv')
    df.to_csv(file, index=False)
    return data(file)
    
def impute_missing_values(df):
    for i in df.columns[df.isnull().any(axis=0)]:
        df[i].fillna(df[i].mean(),inplace=True)
    for col in df.columns[df.eq('?').any()]:
        df[col] =df[col].replace('?', np.nan)
        df[col] = df[col].astype(float)
        df[col] = df[col].fillna(df[col].mean())
    return df

def label_encoding(df):
    syms = [col for col in df.columns if col.strip()[0].islower() and df[col].dtypes == 'O']
    le = LabelEncoder()
    for sym in syms:
        col = le.fit_transform(df[sym])
        df[sym] = col.copy()

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
    tmp = (col.hi - col.lo)/(w.the['bins'] - 1)
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
            n=n+1
            rule[txt]=None
    if n>0:
        return rule

def samples(t, n=None):
    u=[]
    for i in range(1, (n or len(t))+1):
        u.append(t[random.randint(0, len(t)-1)])
    return u

def gaussian(mu, sd):
    mu, sd = mu or 0, sd or 1
    return mu + sd * math.sqrt(-2*math.log(random.random()))*math.cos(2*math.pi*random.random())

def cliffsDelta(ns1,ns2):
    if len(ns1) > 256:
        ns1 = samples(ns1,128)
    if len(ns2) > 256:
        ns2 = samples(ns2,128)
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
    return abs(lt - gt)/n <= 0.4

def delta(i,other):
    e,y,z = 1E-32, i, other
    return abs(y.mu-z.mu)/((e+y.sd**2/y.n+z.sd**2/z.n)**0.5)

def bootstrap(y0,z0):
    x,y,z,yhat,zhat = q.Num(), q.Num(), q.Num(),[],[]
    for y1 in y0:
        x.add(y1)
        y.add(y1)
    for z1 in z0:
        x.add(z1)
        z.add(z1)
    xmu,ymu,zmu = x.mu, y.mu, z.mu
    for y1 in y0:
        yhat.append(y1-ymu+xmu)
    for z1 in z0:
        zhat.append(z1-zmu+xmu)
    tobs = delta(y,z)
    n=0
    for _ in range(1,512+1):
        if delta(q.Num(t=samples(yhat)), q.Num(t=samples(zhat))) > tobs:
            n += 1
    return (n/512) >= 0.9
    
def scottKnot(rxs):
    def merges(i,j):
        out=RX([],rxs[i]['name'])
        for k in range(i, j+1):
            out=merge(out, rxs[j])
        return out
    def same(lo, cut, hi):
        l = merges(lo, cut)
        r = merges(cut+1, hi)
        cliffsDelta(l['has'], r['has']) and bootstrap(l['has'], r['has'])

    def recurse(lo, hi, rank):
        b4=merges(lo, hi)
        best=0
        cut=None
        for j in range(lo, hi+1):
            if j<hi:
                l=merges(lo, j)
                r=merges(j+1, hi)
                now=(l['n']*(mid(l)-mid(b4))**2 + r['n']*(mid(r)-mid(b4))**2) / (l['n']+r['n']) 
                if now>best:
                    if abs(mid(l)-mid(r))>=cohen:
                        cut, best= j ,now
        if cut is not None and not same(lo,cut,hi):
            rank = recurse(lo, cut, rank) + 1
            rank = recurse(cut+1, hi, rank) 
        else:
            for i in range(lo,hi+1):
                rxs[i]['rank'] = rank
        return rank
    for i,x in enumerate(rxs):
        for j,y in enumerate(rxs):
            if mid(x) < mid(y):
                rxs[j],rxs[i]=rxs[i],rxs[j]
    cohen = div(merges(0,len(rxs)-1)) * float(w.the['cohen'])
    recurse(0, len(rxs)-1, 1)
    return rxs

def tiles(rxs):
    huge=float('-inf')
    lo, hi= float('inf'), float('-inf')
    for rx in rxs:
        lo, hi= min(lo, rx['has'][0]), max(hi, rx['has'][len(rx['has'])-1])
    for rx in rxs:
        t, u= rx['has'],[]
        def of(x,most):
            return int(max(1, min(most,x)))
        def at(x):
            return t[of(len(t)*x//1, len(t))]
        def pos(x):
            wid=int(w.the['width'])
            return math.floor(of(wid * (x - lo) / (hi - lo + 1E-32) // 1, wid))
        for i in range(0, int(w.the['width'])+1):
            u.append(" ")
        a,b,c,d,e=at(0.1),at(0.3),at(0.5),at(0.7),at(0.9)
        A,B,C,D,E=pos(a), pos(b), pos(c), pos(d), pos(e)
        for i in range(A,B+1):
            u[i]="-"
        for i in range(D,E+1):
            u[i]="-"
        u[int(w.the['width'])//2]="|" 
        u[C]="*"
        x=[]
        for i in [a,b,c,d,e]:
            x.append(w.the['Fmt'].format(i))
        rx['show'] = ''.join(u) + str(x)
    return rxs


def run_stats(data, top_table):
    print('MWU and KW significance level: ', the['significance_level']/100)
    all =  top_table['all']
    sway1 = top_table['sway1']
    sway2 = top_table['sway2 (pca)']
    sway3 = top_table['sway3 (agglo)']
    sway4 = top_table['sway4 (kmeans)']
    sways = ['sway1', 'sway2', 'sway3', 'sway4']
    mwu_sways = []
    kw_sways = []

    taxes = {'samp tax1': [], 'samp tax2': [], 'samp tax3': [], 'samp tax4': []}

    for col in data.cols.y:
        sway_avgs = [sway1['avg'][col.txt], sway2['avg'][col.txt], sway3['avg'][col.txt], sway4['avg'][col.txt]]

        for idx in range(len(xpln_avgs)):
            taxes['samp tax' + str(idx + 1)].append(round(all['avg'][col.txt] - sway_avgs[idx], 2))

        if col.w == -1:
            best_avg = min(sway_avgs)
        else:
            best_avg = max(sway_avgs)
        best_sway = sways[sway_avgs.index(best_avg)]

        for best in sway1['data']:
            sway1_col = [row.cells[col.at] for row in best.rows]
        for best in sway2['data']:
            sway2_col = [row.cells[col.at] for row in best.rows]
        for best in sway3['data']:
            sway3_col = [row.cells[col.at] for row in best.rows]
        for best in sway4['data']:
            sway4_col = [row.cells[col.at] for row in best.rows]

        _, p_value_sways = kruskal(sway1_col, sway2_col, sway3_col, sway4_col)
        kw_sway_p_values.append(p_value_sways/100)
        # if p_value < the['kw_significance']:
        #     top_table['kw_significant'].append('yes')
        # else:
        #     top_table['kw_significant'].append('no')
        
        groups = [sway1_col, sway2_col, sway3_col, sway4_col]
        num_groups = len(groups)
        p_values_mwu = np.zeros((num_groups, num_groups))
        p_values_kruskal = np.zeros((num_groups, num_groups))

        for i in range(num_groups):
            for j in range(i+1, num_groups):
                _, p = mannwhitneyu(groups[i], groups[j], alternative='two-sided')
                p_values_mwu[i, j] = p
                p_values_mwu[j, i] = p
                _, p_value_kruskal = kruskal(sway1_col, sway2_col, sway3_col, sway4_col)
                p_values_kruskal[i, j] = p_value_kruskal
                p_values_kruskal[j, i] = p_value_kruskal

        # print('Pairwise Mann-Whitney U tests')
        # print(pd.DataFrame(p_values_mwu, index=sways, columns=sways))
        # print()

        # Apply Bonferroni correction for multiple comparisons
        adjusted_p_values = multipletests(p_values_mwu.ravel(), method='fdr_bh')[1].reshape(p_values_mwu.shape)
        post_hoc = pd.DataFrame(adjusted_p_values, index=sways, columns=sways)

        # print("Pairwise Mann-Whitney U tests with Benjamini/Hochberg correction:")
        # print(post_hoc)
        # print()

        krusal_df = pd.DataFrame(p_values_kruskal, index=sways, columns=sways)
        # print("Pairwise Kruskal Wallis:")
        # print(krusal_df)
        # print()

        mwu_sig = set(post_hoc.iloc[list(np.where(post_hoc >= the['significance_level'])[0])].index)
        if len(mwu_sig) == 0:
            mwu_sways.append([best_sway])
        else:
            mwu_sways.append(list(mwu_sig))

        kw_sig = set(krusal_df.iloc[list(np.where(krusal_df >= the['significance_level'])[0])].index)
        if len(kw_sig) == 0:
            kw_sways.append([best_sway])
        else:
            kw_sways.append(list(kw_sig))
    return mwu_sways, kw_sways, taxes, kw_sway_p_values

def get_avgs_from_data_list(data_obj_list):
    avgs = {}
    # For each data_obj, get sum of mid/mode
    for data_obj in data_obj_list:
        for k,v in data_obj.stats().items():
            avgs[k] = avgs.get(k,0) + v
    # Convert sums to averages
    for k,v in avgs.items():
        avgs[k] = round(avgs[k]/20, 2)
    return avgs
