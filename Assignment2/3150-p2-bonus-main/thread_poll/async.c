
#include <stdlib.h>
#include <pthread.h>
#include "async.h"
#include "utlist.h"

my_queue_t *thread_pool;
pthread_t *threads;
int *thread_ids;
pthread_mutex_t count_mutex;

// 循环运行，不停的查找公共queue内是否有空余的任务，若有，则处理；若无则自杀。
void *thread_run( void *t ){
    pthread_mutex_lock(&count_mutex);
    // If there is task in the queue, the thread will handle it/
    if(thread_pool->size != 0){
        int args = thread_pool->tail->args;

        pthread_mutex_unlock(&count_mutex);
        hanlder(args);

    }
    else{
        pthread_mutex_unlock(&count_mutex);
    }
    
}

void async_init(int num_threads) {
    /** TODO: create num_threads threads and initialize the thread pool **/
    // Initialize the lock.
    pthread_mutex_init(&count_mutex, NULL);
    // Allocate Memory
    task_list = (my_queue_t *)malloc(sizeof(my_queue_t));
    task_list->size = 0;
    threads = (pthread_t *)malloc(sizeof(pthread_t) * num_threads);
    thread_ids = (int *)malloc(sizeof(int) * num_threads);
    // Set default thread ids.
    for(int i = 0; i < num_threads; i++) thread_ids[i] = i;
    // Create threads.
    for(int i = 0; i < num_threads; i++){
        pthread_create(&threads[i],&attr, thread_run, (void*)&thread_ids[i]);
    }
    return;
}

void async_run(void (*hanlder)(int), int args) {
    /** TODO: rewrite it to support thread pool **/
    // Throw the task into the task queue.
    pthread_mutex_lock(&count_mutex);
    
    pthread_mutex_unlock(&count_mutex);
    hanlder(args);
}