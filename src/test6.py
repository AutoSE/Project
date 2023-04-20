import cli as c
import num as n
import sym as s
import Utils as u
import os
import Data as d
from tabulate import tabulate
egs={}
line=0

def the():
    print(c.the)
    egs['the'] = c.the
    return True

def rand():
    Seed = 1
    t=[]
    for i in range(1,1000+1):
        t.append(u.rint(0,100,1))
    Seed=1
    U=[]
    for i in range(1,1000+1):
        U.append(u.rint(0,100,1))
    for k,v in enumerate(t):
        assert(v==U[k])

def some():
    c.the['Max'] = 32
    num1 = n.Num()
    for i in range(1,10001):
        num1.add(i)
    print(num1.has)

def nums():
    num1, num2 = n.Num(),n.Num()
    global Seed
    Seed = c.the['seed']
    for i in range(1,10001):
        num1.add(u.rand(0,1))
    Seed = c.the['seed']
    for i in range(1,10001):
        num2.add(u.rand(0,1)**2)
    m1,m2 = u.rnd(num1.mid(),1), u.rnd(num2.mid(),1)
    d1,d2 = u.rnd(num1.div(),1), u.rnd(num2.div(),1)
    print(1, m1, d1)
    print(2, m2, d2) 
    return m1 > m2 and .5 == u.rnd(m1,1)

def syms():
    sym_obj = s.Sym()
    for k,x in enumerate(["a","a","a","a","b","b","c"]):
        sym_obj.add(x)
    print(sym_obj.mid(), u.rnd(sym_obj.div()))
    return 1.379 == u.rnd(sym_obj.div())
'''
def csv():
    def fn(t):
        global numberofchars
        numberofchars+=len(t)
    global numberofchars
    numberofchars=0
    u.csv(c.the['file'],fn)
    return 3192==numberofchars
'''
def data():
    data = d.Data(c.the["file"])
    col = data.cols.x[1]
    print(col.lo,col.hi, col.mid(), col.div())
    print(data.stats('mid', data.cols.y,2))

def clone():
    data1 = d.Data(c.the["file"])
    data2 = data1.clone(data1.rows)
    print(data1.stats('mid', data1.cols.y,2))
    print(data2.stats('mid', data2.cols.y,2))

def cliffsDelta():
    assert(False == u.cliffsDelta([8,7,6,2,5,8,7,3],[8,7,6,2,5,8,7,3]))
    assert(True  == u.cliffsDelta([8,7,6,2,5,8,7,3], [9,9,7,8,10,9,6])) 
    t1,t2=[],[]
    for i in range(1,1001):
        t1.append(u.rand(0,1))
    for i in range(1,1001):
        t2.append(u.rand(0,1)**.5)
    assert(False == u.cliffsDelta(t1,t1))
    assert(True  == u.cliffsDelta(t1,t2))
    diff,j=False,1.0
    while not diff:
        def function(x):
            return x*j
        t3=list(map(function, t1))
        diff=u.cliffsDelta(t1,t3)
        print(">",u.rnd(j),diff)
        j=j*1.025

def dist():
    data = d.Data(c.the['file'])
    num  = n.Num()
    for row in data.rows:
        num.add(data.dist(row, data.rows[1]))
    print({'lo' : num.lo, 'hi' : num.hi, 'mid' : u.rnd(num.mid()), 'div' : u.rnd(num.div())})

def half():
    data=d.Data(c.the['file'])
    left,right,A,B,g,_ = data.half() 
    print(len(left),len(right))
    l,r = data.clone(left), data.clone(right)
    print("l",l.stats('mid', l.cols.y, 2))
    print("r",r.stats('mid', r.cols.y, 2))

def tree():
    data=d.Data(c.the['file'])
    u.showTree(data.tree(),"mid", data.cols.y,1)

def sway():
    data = d.Data(c.the['file'])
    best, rest, _ = data.sway()
    print("\nall ", data.stats('mid', data.cols.y, 2))
    print("    ", data.stats('div', data.cols.y, 2))
    print("\nbest",best.stats('mid', best.cols.y, 2))
    print("    ", best.stats('div', best.cols.y, 2))
    print("\nrest", rest.stats('mid', rest.cols.y, 2))
    print("    ", rest.stats('div', rest.cols.y, 2))

def bins():
    b4=''
    data = d.Data(c.the['file'])
    best, rest, _ = data.sway()
    print("all","","","",{'best':len(best.rows), 'rest':len(rest.rows)})
    for k,t in enumerate(u.bins(data.cols.x,{'best':best.rows, 'rest':rest.rows})):
        for range in t:
            if range['txt'] != b4:
                print("")
            b4 = range['txt']
            print(range['txt'],range['lo'],range['hi'],u.rnd(u.value(range['y'].has, len(best.rows),len(rest.rows),"best")), range['y'].has)

def xpln():
    data = d.Data(c.the['file'])
    best, rest, evals= data.sway()
    rule, most= data.xpln( best, rest)
    if rule:
        print("\n-----------\nexplain=", data.showRule(rule))
        selects=data.selects(rule,data.rows)
        datasels = [s for s in selects if s!=None]
        data1= data.clone(datasels)
        print("all               ",data.stats('mid', data.cols.y, 2),data.stats('div', data.cols.y, 2))
        print("sway with",evals,"evals",best.stats('mid', best.cols.y, 2),best.stats('div', best.cols.y, 2))
        print("xpln on",evals,"evals",data1.stats('mid', data1.cols.y, 2),data1.stats('div', data1.cols.y, 2))
        top,_ = data.betters(len(best.rows))
        top = data.clone(top)
        print("sort with",len(data.rows),"evals",top.stats('mid', top.cols.y, 2),top.stats('div', top.cols.y, 2))

def sway1():
    data = d.Data(c.the['file'])
    data2 = u.preprocess_data(c.the['file'], d.Data)

    best, rest, _ = data2.sway1()
    print("\nall ", data2.stats('mid', data.cols.y, 2))
    print("    ", data2.stats('div', data.cols.y, 2))
    print("\nbest",best.stats('mid', best.cols.y, 2))
    print("    ", best.stats('div', best.cols.y, 2))
    print("\nrest", rest.stats('mid', rest.cols.y, 2))
    print("    ", rest.stats('div', rest.cols.y, 2))




def sway2():
    data = d.Data(c.the['file'])
    data2 = u.preprocess_data(c.the['file'], d.Data)

    best, rest, _ = data2.sway2()
    print("\nall ", data.stats('mid', data.cols.y, 2))
    print("    ", data.stats('div', data.cols.y, 2))
    print("\nbest",best.stats('mid', best.cols.y, 2))
    print("    ", best.stats('div', best.cols.y, 2))
    print("\nrest", rest.stats('mid', rest.cols.y, 2))
    print("    ", rest.stats('div', rest.cols.y, 2))

def table():
    top_table = {'all': {'data' : [], 'evals' : 0},
             'sway': {'data' : [], 'evals' : 0},
             'sway1': {'data' : [], 'evals' : 0},
             #'sway2': {'data' : [], 'evals' : 0},
             'top': {'data' : [], 'evals' : 0}}

    bottom_table = [[['all', 'all'],None],
                [['all', 'sway'],None],
                [['sway', 'sway1'],None],
                [['sway', 'sway2'],None],
                [['sway1', 'sway2'],None],
                [['sway1', 'top'],None]]
    count = 0
    while count < 20:
        data = d.Data(c.the['file'])
        best, rest, evals = data.sway()
        rule, most = data.xpln(best, rest)
        print('------------------------------------------------------')
        print(best,rest,evals,rule,most)
        print('------------------------------------------------------')
        data = d.Data(c.the['file'])
        data2 = u.preprocess_data(c.the['file'], d.Data)
        best1, rest1, evals1 = data2.sway1()
        #best2, rest2, evals2 = data2.sway2()
        if rule!=-1:
            betters, _ = data.betters(len(best.rows))
            top_table['top']['data'].append(d.Data(c.the['file']))
            top_table['all']['data'].append(data)
            top_table['sway']['data'].append(best)
            top_table['sway1']['data'].append(best1)
            #top_table['sway2']['data'].append(best2)
            top_table['all']['evals'] += 0
            top_table['sway']['evals'] += evals
            top_table['sway1']['evals'] += evals1
            #top_table['sway2']['evals'] += evals2
            top_table['top']['evals'] += len(data.rows)
            '''for i in range(len(bottom_table)):
                [base, diff], result = bottom_table[i]
                if result == None:
                    bottom_table[i][1] = ['=' for _ in range(len(data.cols.y))]
                for k in range(len(data.cols.y)):
                    if bottom_table[i][1][k] == '=':
                        y0, z0 = top_table[base]['data'][count].cols.y[k],top_table[diff]['data'][count].cols.y[k]
                        is_equal = u.bootstrap(y0.vals(), z0.vals()) and u.cliffsDelta(y0.vals(), z0.vals())
                        if not is_equal:
                            bottom_table[i][1][k] = '≠'''
            count+=1

    headers = [y.txt for y in data.cols.y]
    table = []

    top_table['sway'] = top_table.pop('sway')
    top_table['sway1'] = top_table.pop('sway1')
    #top_table['sway2'] = top_table.pop('sway2')
    top_table['top'] = top_table.pop('top')
    #print(top_table)   #added this print,delete later



    for k,v in top_table.items():
        print(k,v)
        v['avg'] = u.get_avgs_from_data_list(v['data'])
        print(v['avg'])
        stat = [k] + [v['avg'][y] for y in headers]
        stat += [int(v['evals']/20)]
        table.append(stat)
    '''mwu_sways, kw_sways, taxes, kw_sway_p_values = u.run_stats(data2, top_table)
    for k, v in taxes.items():
        table.append([k] + v)
        table.append(['KW Sway p-vals'] + kw_sway_p_values)
        table.append(['Mann-Whitney U Sways'] + mwu_sways)
        table.append(['Kruskal-Wallis Sways'] + kw_sways)'''
    print(tabulate(table, headers=headers+["n_evals avg"],numalign="right"))
    print()

def all():
    print('the')
    egs['the']=the()
    print('rand')
    egs['rand']=rand()
    print('some')
    egs['some']=some()
    print('nums')
    egs['nums']=nums()
    print('syms')
    egs['syms']=syms()
    #print('csv')
    #egs['csv']=csv()
    print('data')
    egs['data']=data()
    print('clone')
    egs['clone']=clone()
    print('dist')
    egs['dist']=dist()
    print('half')
    egs['half']=half()
    print('tree')
    egs['tree']=tree()
    print('sway')
    egs['sway']=sway()
    print('sway1')
    egs['sway1']=sway1()
    print('sway2')
    egs['sway2']=sway2()
    print('bins')
    egs['bins']=bins()
    print('xpln')
    egs['xpln']=xpln()
    print('table')
    egs['table']=table()

