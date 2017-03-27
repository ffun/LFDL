'''
file:Checker.py\n
@author:fang.junpeng\n
@email:tfzsll@126.com
'''
import sys
class Checker(object):
    '''
    class for assert value
    '''
    @classmethod
    def failed_and_errout(cls, BoolValue, msg):
        '''
        check value and output when False
        '''
        if not BoolValue:
            sys.stderr.write(msg)
            assert BoolValue

    '''
    Class for check something
    '''
    @classmethod
    def type_check(cls, obj, types, msg="Err"):
        '''
        check type\n
        @obj:object\n
        @types:all types which could be accept
        '''
        Good = False
        for i in xrange(len(types)):
            if isinstance(obj, types[i]):
                Good = True
                break
        cls.failed_and_errout(Good, msg)
    @classmethod
    def seq_len_check(cls, obj, length, msg='Err'):
        '''
        seq length check
        '''
        cls.failed_and_errout(len(obj) == length, msg)
    @classmethod
    def num_check(cls, num1, num2, msg='Err'):
        cls.failed_and_errout(num1 == num2, msg)
