'''
file:Checker.py
@author:fang.junpeng
@email:tfzsll@126.com
'''
class Checker(object):
    '''
    Class for check something
    '''
    @classmethod
    def type_check(cls, obj, types):
        '''
        check type\n
        @
        '''
        Good = False
        for i in range(len(types)):
            if obj is types[i]:
                Good = True
                break
        assert Good is True
    @classmethod
    def seq_len_check(cls, obj, length):
        '''
        seq length check
        '''
        assert len(obj) == length
