import multiprocessing
import multiprocessing.connection

def f(conn: multiprocessing.connection.Connection):
    while conn.readable:
        print('robot received information: ' + str(conn.recv())[: 64] + ' ...')