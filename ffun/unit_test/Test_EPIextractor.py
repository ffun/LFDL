import ffun.io as Fio

#example:extract epi file,return a numpy ndarray
extractor = Fio.EPIextractor('/Users/fang/workspaces/tf_space/Box/epi45_53/epi_45_53_002.png')
extractor.save_extract(256,'/Users/fang/workspaces/tf_space/test')