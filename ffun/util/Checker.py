'''
file:Checker.py\n
@author:fang.junpeng\n
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
        @obj:object\n
        @types:all types which could be accept
        '''
        Good = False
        for i in range(len(types)):
            if isinstance(obj,types[i]):
                Good = True
                break
        assert Good is True
    @classmethod
    def seq_len_check(cls, obj, length):
        '''
        seq length check
        '''
        assert len(obj) == length
