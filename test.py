#!/usr/bin/python
# -*- coding: UTF-8 -*-

import ffun.io as Fio

#files =  Fio.FolderHelp.get_files('./pngdata')
Epi_creator = Fio.EPIcreator('/Users/fang/workspaces/tf_space/LFDL/pngdata')
Epi_creator.create((36,44))