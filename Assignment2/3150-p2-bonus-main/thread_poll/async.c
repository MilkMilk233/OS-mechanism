
#include <stdlib.h>
#include <pthread.h>
#include <stdio.h>
#include "async.h"
#include "utlist.h"

my_queue_t *task_list;
pthread_t *threads;
int *thread_ids;
pthread_mutex_t count_mutex;
pthread_attr_t attr;

// Keep running, every time check if the queue is empty, if yes, skip it. Otherwise, handle it.
void *thread_run( void *t ){
    while(1){
        pthread_mutex_lock(&count_mutex);
        // If there is task in the queue, the thread will handle it/
        if(task_list->size != 0){
            int args = task_list->tail->args;
            // Discard the tail task, free the space.
            my_item_t *target_item = task_list->tail;
            void (*handler)(int) = target_item->handler;
            if(task_list->size == 1){
                task_list->size = 0;
                task_list->head = NULL;
                task_list->tail = NULL;
            }
            else{
                task_list->tail = target_item->prev;
                task_list->size--;
            }
            free(target_item);
            pthread_mutex_unlock(&count_mutex);
            // Handle that task!
            handler(args);

        }
        else{
            pthread_mutex_unlock(&count_mutex);
        }
    }
    printf("Process quit.\n");
    pthread_exit(NULL);
}

void async_init(int num_threads) {
    /** TODO: create num_threads threads and initialize the thread pool **/
    // Initialize the lock.
    pthread_mutex_init(&count_mutex, NULL);
    pthread_attr_init(&attr);
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
    my_item_t *item = (my_item_t *)malloc(sizeof(my_item_t));
    item->handler = hanlder;
    item->args = args;
    item->next = NULL;
    pthread_mutex_lock(&count_mutex);
    if(task_list->size != 0){
        item->prev = task_list->tail;
    }
    else{
        item->prev = NULL;
        task_list->head = item;
    }
    task_list->size++;
    task_list->tail = item;
    pthread_mutex_unlock(&count_mutex);
}