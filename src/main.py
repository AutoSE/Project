import Utils as u
import test6 as t
import cli as c
import copy
import sys
def main():
    t.all()
    saved, fails = copy.deepcopy(c.the), 0
    for k,v in c.cli(u.settings(c.help)).items():
        c.the[k]=v
        saved[k]=v
    if c.the['help']== True:
        print(c.help)
    else:
        for what, fun in t.egs.items():
            if c.the['go']=='all' or c.the['go']==what:
                for k, v in saved.items():
                    c.the[k]=v
                u.Seed=c.the["seed"]
            if t.egs[what]==False:
                fails+=1
                print('XX FAIL:', what)
            else:
                print("✅ PASS:", what)
    sys.exit(fails)
if __name__=='__main__':
    main()