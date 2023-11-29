# THREAD POOL

## Task
* Implement in async.c and async.h: ```void async_init(int num_threads)``` and ```void async_run(void (*handler)(int), int args)```
* You can use list data structure in utlist.h, for example: ```DL_APPEND(my_queue->head, my_item);```(adding to queue end) and  ```DL_DELETE(my_queue->head, my_queue->head);```(popping from queue head)
* When no jobs are coming, your threads created in ```async_init``` have to go to sleep and is not allowed to do busy waiting like ```while(1){sleep(any);}```, and when jobs are coming a sleeping thread in your thread pool **must** wake up immediately (that is, no ```sleep()``` call is allowed).
* async_run should be asynchronous without further call to pthread_create, that is it should return immediately before the job is handled (in the code we give you, async_run runs synchronously, so you need to rewrite the function)

## Test
* ./httpserver --proxy inst.eecs.berkeley.edu:80 --port 8000 --num-threads 5
* When you run the test, you can access 127.0.0.1:8000 at your browser even before modifying the code, but it cannot serve the request concurrently (multiple requests at the same time) and always serve with the same thread id. After implementing the thread pool you should support concurrent access.
* Once you create num-threads of threads in ```async_init``` to initialize your pool, you are not allowed to create any more thread in ```async_run```, otherwise zero score will be granted.

## Solution

#### Too long don't read

I implemented thread pool by **producer-consumer pattern**.

- There is a waiting list, all the pending tasks are stored in it. 
- Each thread is the **consumer**. They execute the remaining tasks in the waiting list one after another. If **consumer** has no task to execute, it will go to sleep.
- `async_run` is **producer**. It will pass the task into the waiting list.  After **producer** pass in a new task, it will randomly wake one **consumer** up. 

#### Waiting list

I didn't used the data structure provided. Instead, I designed it on my own. I implemented an queue structure, which is easy for me to insert at the head and delete from the tail. For each consumer, it will produces one task. At that time, the task will be packed as the structure 'item' and throw it into the tail of the queue. Then the `async_run()` will return immediately, making it a asynchronized function (Notice that it is called asynchronized because the `async_run()` only need to throw the task into the waiting list, but not to directly calculate it). Then, the hungry consumers will fetch these task from waiting list, and delete them from the waiting list..

#### Avoid keep spinning

At the first time I implement it, I made a mistake. I let those hungry consumers keep searching if there is anything in the waiting list. The potential problem is, when no request coming in, these threads still spinning over and over again, which brings a log of burden for the system resources (It takes up ~100% of my CPU resources!). The ideal case is, when there is no task coming in, the consumer threads should fall asleep. Next time a new task is added to the waiting list (i.e. the `async_run` is called), the consumer threads should wake up and handle them. So I made some important changes: to add the conditional variable to wake those threads up. For each thread, if it sees no tasks in the waiting list, it will wait for signals in conditional variable, meanwhile to release the mutually exclusive lock. Then after several CPU cycles, all the threads will fall asleep. Next time the tasks crowded in, the `async_run` call the `pthread_cond_signal` to randomly pick one thread to wake up, the lucky strike being selected will re-enter the endless loop. With more and more tasks flourish in, the threads are wake up one after another, until all of them get to work. This mechanism can greatly reduce the cost of the system resources when no tasks for a long time, and Increase system persistence.

## How to run

```bash
cd ~/async_thread_poll/thread_poll/
make
./httpserver --files files/ --port 8000 --num-threads 10
```

Open up a new terminal

```bash
ab -n 5000 -c 10 http://localhost:8000/