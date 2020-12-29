num_users = 3773
num_items = 4544
# num_users = 30887
# num_items = 18995
f_n = './data/beibei/beibei_all.links'
with open(f_n) as f:
    x = [[int(s.split('\t')[0]), int(s.split('\t')[1])] for s in f.readlines()]
print(len(x))
y = []
for s in x:
    if s[0] < s[1]:
        y.append([s[0], s[1]])
    else:
        y.append([s[1], s[0]])
y = np.unique(y, axis=0)
print(len(y))
with open(f_n, 'w') as f:
    for yy in y:
        print >> f, str(yy[0]) + '\t' + str(yy[1])