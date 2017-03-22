
import ffun.util as Fut
import ffun.io as Fio

files =  Fio.FileHelper.get_files('/Users/fang/workspaces/tf_space/box')
Epi_creator = Fio.EPIcreator(files)
Epi_creator.create((45,53))