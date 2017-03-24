import ffun.io as Fio

#get labels,return a matrix of 512x512
#every element of the matrix is a float
labelloader = Fio.TextLoader()
data = labelloader.read('/Users/fang/workspaces/tf_space/LFDL/disp.txt', float)
print len(data)