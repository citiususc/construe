# axel.py
#
# Copyright (C) 2016 Adrian Cristea adrian dot cristea at gmail dotcom
#
# Based on an idea by Peter Thatcher, found on
# http://www.valuedlessons.com/2008/04/events-in-python.html
#
# This module is part of axel and is released under
# the MIT License: http://www.opensource.org/licenses/mit-license.php 
#
# Contributors:
#  Erwin Mayer <traderwin at gmail dot com>
#  Rob van der Most <Rob at rmsoft dot nl>
#
# Source: http://pypi.python.org/pypi/axel
# Docs:   http://packages.python.org/axel

import sys
from threading import Thread, RLock

try:
    from Queue import Queue, Empty
except ImportError:
    from queue import Queue, Empty


class Event(object):
    """ 
    Event object inspired by C# events. Handlers can be registered and
    unregistered using += and -= operators. Execution and result are  
    influenced by the arguments passed to the constructor and += method.
        
    from axel import Event
     
    event = Event()
    def on_event(*args, **kw):
        return (args, kw)
    
    event += on_event                        # handler registration
    print(event(10, 20, y=30))        
    >> ((True, ((10, 20), {'y': 30}), <function on_event at 0x00BAA270>),)
     
    event -= on_event                        # handler is unregistered
    print(event(10, 20, y=30))     
    >> None
     
    class Mouse(object):
        def __init__(self):
            self.click = Event(self)
            self.click += self.on_click      # handler registration
    
        def on_click(self, sender, *args, **kw):
            assert isinstance(sender, Mouse), 'Wrong sender'
            return (args, kw)
    
    mouse = Mouse()
    print(mouse.click(10, 20))        
    >> ((True, ((10, 20), {}), 
    >>  <bound method Mouse.on_click of <__main__.Mouse object at 0x00B6F470>>),)
    
    mouse.click -= mouse.on_click            # handler is unregistered
    print(mouse.click(10, 20))                        
    >> None
    """

    def __init__(self, sender=None, asynch=False, exc_info=False,
                 threads=3, traceback=False):
        """ Creates an event 
        
        asynch
            if True, handlers are executed asynchronously (the main thread is not 
            waiting for handlers execution to complete). In this case, no data
            regarding the success and result of the execution is returned.
        exc_info
            if True, result will contain sys.exc_info()[:2] on error
        sender
            event's sender. The sender is passed as the first argument to the 
            handler, only if is not None. For this case the handler must have
            a placeholder in the arguments to receive the sender
        threads
            maximum number of threads that will be started for handlers execution             
            threads = 0: 
                - synchronized execution in the main thread
                - asynch must be False 
            threads = 1: 
                - synchronized execution in separate thread
                - asynch can be True or False 
            threads > 1: 
                - desynchronized execution in separate threads
                - asynch can be True or False              
            visual representation:
                threads = 0:
                    [Main Thread]: event (handler1, handler2, handler3, handler4)
                    [Main Thread]: ..handler1, handler2, handler3, handler4
                threads = 1:
                    [Main Thread]: event (handler1, handler2, handler3, handler4)
                       [Thread-1]: ..handler1, handler2, handler3, handler4
                threads > 1 (let's say 2): (thread switch is not accurate)
                    [Main Thread]: event (handler1, handler2, handler3, handler4)
                       [Thread-1]: ..handler1 <= assume short running
                       [Thread-2]: ..handler2 <= assume long running 
                       [Thread-1]: ..handler3 <= assume short running
                       [Thread-1]: ..handler4 <= assume short running
        traceback
            if True, the execution result will contain sys.exc_info() 
            on error. exc_info must be also True to get the traceback      
                
        hash = hash(handler)
        
        Handlers are stored in a dictionary that has as keys the handler's hash        
            handlers = { 
                hash : (handler, memoize, timeout),
                hash : (handler, memoize, timeout), ... 
            }
        The execution result is cached using the following structure    
            memoize = { 
                hash : ((args, kw, result), (args, kw, result), ...), 
                hash : ((args, kw, result), ...), ...                      
            } 
        The execution result is returned as a tuple having this structure
            exec_result = (
                (True, result, handler),        # on success
                (False, error_info, handler),   # on error 
                (None, None, handler), ...      # asynchronous execution
            )
        """
        if asynch and threads == 0:
            raise ValueError('Asynch execution is only possible if threads > 0')
        self.asynch = bool(asynch)
        self.exc_info = bool(exc_info)
        self.sender = sender
        self.threads = int(threads)
        self.traceback = bool(traceback)
        self.handlers = {}
        self.memoize = {}
        self._hlock = RLock()
        self._mlock = RLock()

    def handle(self, handler):
        """ Registers a handler. The handler can be transmitted together 
        with two arguments as a list or dictionary. The arguments are:
        
        memoize 
            if True, the execution result will be cached in self.memoize
        timeout 
            will allocate a predefined time interval for the execution
            
        If arguments are provided as a list, they are considered to have 
        this sequence: (handler, memoize, timeout)                
        
        Examples:
            event += handler    
            event += (handler, True, 1.5)
            event += {'handler':handler, 'memoize':True, 'timeout':1.5}         
        """
        handler_, memoize, timeout = self._extract(handler)
        with self._hlock:
            handlers = self.handlers.copy()  # Immutable as in .NET http://stackoverflow.com/a/786455/541420
            handlers[hash(handler_)] = (handler_, memoize, timeout)
            self.handlers = handlers
        return self

    def unhandle(self, handler):
        """ Unregisters a handler """
        h, _, _ = self._extract(handler)
        key = hash(h)
        with self._hlock:
            if key not in self.handlers:
                raise ValueError('Handler "%s" was not found' % str(h))
            handlers = self.handlers.copy()
            del handlers[key]
            self.handlers = handlers
        return self

    def fire(self, *args, **kw):
        """ Stores all registered handlers in a queue for processing """
        result = []

        with self._hlock:
            handlers = self.handlers

        if self.threads == 0:  # same-thread execution - synchronized
            for k in handlers:
                # handler, memoize, timeout
                h, m, t = handlers[k]
                try:
                    r = self._memoize(h, m, t, *args, **kw)
                    result.append(tuple(r))
                except:
                    result.append((False, self._error(sys.exc_info()), h))

        elif self.threads > 0:  # multi-thread execution - desynchronized if self.threads > 1
            queue = Queue()

            # result lock just in case [].append() is not  
            # thread-safe in other Python implementations
            rlock = RLock()

            def _execute(*args, **kw):
                """ Executes all handlers stored in the queue """
                while True:
                    try:
                        item = queue.get()
                        if item is None:
                            queue.task_done()
                            break

                        # handler, memoize, timeout
                        h, m, t = handlers[item]  # call under active lock

                        try:
                            r = self._memoize(h, m, t, *args, **kw)
                            if not self.asynch:
                                with rlock:
                                    result.append(tuple(r))
                        except:
                            if not self.asynch:
                                with rlock:
                                    result.append((False, self._error(sys.exc_info()), h))

                        queue.task_done()

                    except Empty:  # never triggered, just to be safe
                        break

            if handlers:
                threads = self._threads(handlers=handlers)

                for _ in range(threads):
                    t = Thread(target=_execute, args=args, kwargs=kw)
                    t.daemon = True
                    t.start()

                for k in handlers:
                    queue.put(k)

                    if self.asynch:  # main thread, no locking required
                        h, _, _ = handlers[k]
                        result.append((None, None, h))

                for _ in range(threads):
                    queue.put(None)  # stop each worker

                if not self.asynch:
                    queue.join()

        return tuple(result) or None

    def count(self):
        """ Returns the count of registered handlers """
        with self._hlock:
            return len(self.handlers)

    def clear(self):
        """ Discards all registered handlers and cached results """
        with self._hlock:
            self.handlers.clear()
        with self._mlock:
            self.memoize.clear()

    def _extract(self, item):
        """ Extracts a handler and handler's arguments that can be provided 
        as list or dictionary. If arguments are provided as list, they are 
        considered to have this sequence: (handler, memoize, timeout)                
        Examples:
            event += handler
            event += (handler, True, 1.5)
            event += {'handler':handler, 'memoize':True, 'timeout':1.5}
        """
        if not item:
            raise ValueError('Invalid arguments')

        handler = None
        memoize = False
        timeout = 0

        if not isinstance(item, (list, tuple, dict)):
            handler = item
        elif isinstance(item, (list, tuple)):
            if len(item) == 3:
                handler, memoize, timeout = item
            elif len(item) == 2:
                handler, memoize = item
            elif len(item) == 1:
                handler = item
        elif isinstance(item, dict):
            handler = item.get('handler')
            memoize = item.get('memoize', False)
            timeout = item.get('timeout', 0)
        return handler, bool(memoize), float(timeout)

    def _memoize(self, handler, memoize, timeout, *args, **kw):
        """ Caches the execution result of successful executions        
        hash = hash(handler)
        memoize = { 
            hash : ((args, kw, result), (args, kw, result), ...), 
            hash : ((args, kw, result), ...), ...                       
        }
        """
        if not isinstance(handler, Event) and self.sender is not None:
            args = list(args)[:]
            args.insert(0, self.sender)

        if not memoize:  # no caching

            if timeout <= 0:  # no time restriction
                return True, handler(*args, **kw), handler

            result = self._timeout(timeout, handler, *args, **kw)
            if isinstance(result, tuple) and len(result) == 3:
                if isinstance(result[1], Exception):  # error occurred
                    return False, self._error(result), handler
            return True, result, handler

        else:  # caching
            with self._mlock:  # cache structure lock
                hash_ = hash(handler)
                if hash_ in self.memoize:
                    for args_, kw_, result in self.memoize[hash_]:
                        if args_ == args and kw_ == kw:  # shallow structure comparison only
                            return True, result, handler

                if timeout <= 0:  # no time restriction
                    result = handler(*args, **kw)
                else:
                    result = self._timeout(timeout, handler, *args, **kw)
                    if isinstance(result, tuple) and len(result) == 3:
                        if isinstance(result[1], Exception):  # error occurred
                            return False, self._error(result), handler

                if hash_ not in self.memoize:
                    self.memoize[hash_] = []
                self.memoize[hash_].append((args, kw, result))
                return True, result, handler

    def _timeout(self, timeout, handler, *args, **kw):
        """ Controls the time allocated for the execution of a method """
        t = spawn_thread(target=handler, args=args, kw=kw)
        t.daemon = True
        t.start()
        t.join(timeout)

        if not t.is_alive():
            if t.exc_info:
                return t.exc_info
            return t.result
        else:
            try:
                msg = '[%s] Execution was forcefully terminated'
                raise RuntimeError(msg % t.name)
            except:
                return sys.exc_info()

    def _threads(self, handlers):
        """ Calculates maximum number of threads that will be started """
        if self.threads < len(handlers):
            return self.threads
        return len(handlers)

    def _error(self, exc_info):
        """ Retrieves the error info """
        if self.exc_info:
            if self.traceback:
                return exc_info
            return exc_info[:2]
        return exc_info[1]

    __iadd__ = handle
    __isub__ = unhandle
    __call__ = fire
    __len__ = count


class spawn_thread(Thread):
    """ Spawns a new thread and returns the execution result """

    def __init__(self, target, args=(), kw={}, default=None):
        Thread.__init__(self)
        self._target = target
        self._args = args
        self._kwargs = kw
        self.result = default
        self.exc_info = None

    def run(self):
        try:
            self.result = self._target(*self._args, **self._kwargs)
        except:
            self.exc_info = sys.exc_info()
        finally:
            del self._target, self._args, self._kwargs
