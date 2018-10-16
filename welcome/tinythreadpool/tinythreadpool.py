#!/usr/bin/env python

from threading import RLock, Semaphore, Thread
import time


class PoolThread:
    def __init__( self,thread_id ):
        self.thread         = Thread( target = self.run )
        self.is_running     = False
        self.thread_id      = thread_id
        self.task_queue     = [ ]
        self.lock           = RLock()
        self.sema           = Semaphore(0)
        self.ready_sema     = Semaphore(0)
        self.ready          = True
    def run(self):
#        print ( "acquire" )
        while self.is_running:
#            print ( "acquire" )
            self.sema.acquire()
            with self.lock:
                if not self.is_running:
                    break
                print ( "task %d" % self.thread_id )
                (task, args) = self.task_queue.pop()
            #with self.lock:
#            print ( args )
            task(*args)
            with self.lock:
#                print ( "%d: %d" % (self.thread_id,len(self.task_queue)) )
                if len(self.task_queue) == 0:
                    self.ready_sema.release()
                    self.ready = True
#                    print ( "%d finished" % self.thread_id )
#        print ( "exit" )
        self.sema.acquire(False)
    def start(self):
        with self.lock:
            self.is_running = True
            if not self.thread:
                self.thread = Thread( target = self.run )
            self.thread.start()
    def stop(self):
        self.is_running = False
        self.sema.release()
        self.thread.join()
        self.thread = None
        self.ready = True
    def enqueue_task(self,task,*args):
        with self.lock:
            self.task_queue.append((task,args))
            self.ready = False
            self.sema.release()
    def running(self):
        with self.lock:
            return self.is_running
    def wait_ready(self,block = True):
#        print ( "%d:wait_ready1"% self.thread_id )
        if not self.ready:
            if block:
                self.ready_sema.acquire()
            else:
                return False
#        print ( "wait_ready2" )
        return True
    def __len__(self):
        with self.lock:
            return len(self.task_queue)
                

class TinyThreadPool:
    def __init__(self,num):
        self.numthreads = num
        self.lock = RLock()
        self.threads = [ PoolThread(i) for i in xrange(0,self.numthreads) ]
    def start(self):
        map(PoolThread.start,self.threads)
    def stop(self):
        map(PoolThread.stop,self.threads)
    def enqueue_task(self,task,*args):
        min_load_thread = min([ (len(i),i) for i in self.threads ])[1]
        min_load_thread.enqueue_task(task,*args)
    def enqueue_task_id(self,t,task,*args):
        thread = self.threads[t]
        thread.wait_ready()
        thread.enqueue_task(task,*args)
    def map(self,task,lst):
        for i in lst:
            self.enqueue_task(task,i)
    def wait_ready(self):
        map(PoolThread.wait_ready,self.threads)
        
def func(i):
    for j in i:
        if j[0] & 1:
            time.sleep(1)
            print ( j[0] )
        j[1][j[0]]*=2

class T:
    def __init__(self, N):
      self.N = N
    def pr(self):
      print ( self.N )

if __name__ == "__main__":
    THREAD_NUM = 100
    POOL = TinyThreadPool(THREAD_NUM)
    LST = [ i for i in xrange(0,THREAD_NUM) ]
    POOL.start()
#    POOL.map(func, [ (i,LST) for i in xrange(0,THREAD_NUM) ]  )
    POOL.enqueue_task(T.pr, T(5))
    POOL.enqueue_task(T.pr, T(5))
    POOL.enqueue_task(T.pr, T(5))
    POOL.enqueue_task(T.pr, T(5))
    POOL.enqueue_task(T.pr, T(5))
    POOL.wait_ready()
#    print ( LST )
    POOL.stop()