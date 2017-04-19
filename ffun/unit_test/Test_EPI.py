from EPI import *
from FileHelper import FileHelper
#1th:test PatchHelper
path = '/Users/fang/workspaces/tf_space/test/EPI-u/000.png'
ih = ImageHelper().read(path)
print ih.channels()
ph = PatchHelper(ih.data_convert3d())
#ph.padding([0, 0, 16, 16])
ph.extract([9, 33], [1, 1])
print ph.size()

path_v = '/Users/fang/workspaces/tf_space/test/EPI-v/000.png'
ih = ImageHelper().read(path_v)
print ih.channels()
ph = PatchHelper(ih.data_convert3d())
ph.padding([16, 16, 0, 0])
ph.extract([33, 9], [1, 1])
print ph.size()
# 2th:test EPI
'''
files =  FileHelper.get_files('/Users/fang/workspaces/tf_space/box', '.png')
epi = EPI(files)
epi.create(range(36, 45), 'v','/Users/fang/workspaces/tf_space/test/EPI-v')
'''
