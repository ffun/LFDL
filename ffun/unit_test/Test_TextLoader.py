from FileHelper import LabelHelper

#get labels,return a matrix of 512x512
#every element of the matrix is a float
labelloader = LabelHelper()
data = labelloader.read('/Users/fang/workspaces/tf_space/LFDL/disp.txt', float)
print len(data)