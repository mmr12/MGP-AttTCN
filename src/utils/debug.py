import datetime
import sys
import time


def print_time():
    print(datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'))
    sys.stdout.flush()


def flush_print(string):
    print(string)
    sys.stdout.flush()


def t_print(string):
    T = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
    print(T, " -- ", string)
    sys.stdout.flush()
