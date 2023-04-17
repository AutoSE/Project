'''import sys
import re 
#from codes.utilities import utilities as u
'''
def coerce(s):
    return int(s) if s.isdigit() else s

'''Defines command line arguments and config dictionary'''
'''
#default values
help = "script.lua : an example script with help text and a test suite\n\n\
-d --dump   on crash, dump stack = false\n\
-f  --file  name of file = ../etc/data/auto93.csv\
-g --go start-up action = data\n\
-h --help   show help   = false\n\
-s --seed   random number seed  = 937162211\n\
"
arg = sys.argv[1:]

the = {}
#compile pattern with help and turn into dictionary
pattern = re.compile("\n[-][\S]+[\s]+[-][-]([\S]+)[^\n]+=[\s]([\S]+)")
#pattern = re.compile("\n[\s]+[-][\S]+[\s]+[-][-]([\S]+)[^\n]+= ([\S]+)")
for match in pattern.finditer(help):
    k, v = match.group(1, 2)
    the[k] = coerce(v)

print(the)

"""
Reads in default options and stores in configuration dictionary "the"
t = dictionary of options
"""
def cli(t):
    for slot, v in t.items():
        v = str(v)
        for n, x in enumerate(arg):
            if x == "-" + slot[0] or x == "--" + slot:
                v = v == "false" and "true" or v == "true" and "false" or arg[n + 1]
        t[slot] = coerce(v)
    if len(sys.argv) == 2 and arg[0] == "help":
        print("\n" + help + "\n")
        exit()
    return t     

the = cli(the)
#print(the)


'''



import sys
import re 
#from codes.utilities import utilities as u

'''Defines command line arguments and config dictionary'''

#default values
help = "script.lua : an example script with help text and a test suite\n\n\
-b  --bins    initial number of bins  = 16\n\
-c  --cliffs  cliff's delta threshold = .147\n\
-f --file with csv data = ./etc/data/auto93.csv\n\
-F  --Far     distance to \"faraway\" = .95\n\
-g  --go      start-up action = nothing\n\
-h --help   show help   = false\n\
-H  --Halves  search space for clustering  = 512\n\
-m  --min     stop clusters at N^min = .5\n\
-M  --Max     numbers                = 512\n\
-p  --p       distance coefficient   = 2\n\
-r  --rest    how many of rest to sample   = 4\n\
-s --seed   random number seed  = 937162211\n\
-R  --Reuse   child splits reuse a parent pole = true\n\
-S  --Sample   sampling data size     = 512\n\
"
arg = sys.argv[1:]

the = {}
#compile pattern with help and turn into dictionary
pattern = re.compile("\n[-][\S]+[\s]+[-][-]([\S]+)[^\n]+=[\s]([\S]+)")
for match in pattern.finditer(help):
    k, v = match.group(1, 2)
    the[k] = coerce(v)

"""
Reads in default options and stores in configuration dictionary "the"
t = dictionary of options
"""
def cli(t):
    for slot, v in t.items():
        v = str(v)
        for n, x in enumerate(arg):
            if x == "-" + slot[0] or x == "--" + slot:
                v = v == "false" and "true" or v == "true" and "false" or arg[n + 1]
        t[slot] = coerce(v)
    if len(sys.argv) == 2 and arg[0] == "help":
        print("\n" + help + "\n")
        exit()
    return t     

the = cli(the)
#print(the)