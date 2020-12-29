import random
import setproctitle
import argparse
setproctitle.setproctitle('dataset_part@linian')
def link_partion(name):
    a = set([(int(x.split('\t')[0]), int(x.split('\t')[1])) for x in open('/home/linian/diffnet-master/data/{}/{}_all.links'.format(name, name)).readlines()])
    b = set(random.sample(a, int(len(a)*.7)))
    bb = []
    for [u0, u1] in list(b):
        bb.append(str(u0)+'\t'+str(u1)+'\n')
    f = open('/home/linian/diffnet-master/data/{}/{}.train.links'.format(name, name), 'w+')
    f.writelines(bb)
    f.close()
    
    c = set(random.sample(b, int(len(b)*.667)))
    cc = []
    for [u0, u1] in list(c):
        cc.append(str(u0)+'\t'+str(u1)+'\n')
    f = open('/home/linian/diffnet-master/data/{}/{}.val.links'.format(name, name), 'w+')
    f.writelines(cc)
    f.close()
    
    d = b - c
    dd = []
    for [u0, u1] in list(d):
        dd.append(str(u0)+'\t'+str(u1)+'\n')
    f = open('/home/linian/diffnet-master/data/{}/{}.test.links'.format(name, name), 'w+')
    f.writelines(dd)
    f.close()
    
parser = argparse.ArgumentParser(description='Welcome to the Experiment Platform Entry')
parser.add_argument('--link_name', nargs='?', help='link name')
args = parser.parse_args()

if __name__ == "__main__":
    link_partion(args.link_name)