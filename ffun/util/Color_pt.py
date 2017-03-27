#!/usr/bin/python
# -*- coding: UTF-8 -*-
'''
author:fang.junpeng\n
email:tfzsll@126.com\n
'''
def pt(obj, color='green'):
    color_set = {'red':'0;31m','green':'0;32m','blue':'0;33m'}
    color_cmd = None
    #if has this color,set default color
    if not color_set.has_key(color):
        color = 'green'
    #get color cmd
    color_cmd = '\033['+color_set[color]
    print color_cmd #set color
    color_normal = '\033[0m'
    #print obj
    print obj
    #恢复默认颜色
    if color_set.has_key(color):
        print color_normal
