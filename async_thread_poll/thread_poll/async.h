#ifndef __ASYNC__
#define __ASYNC__

#include <pthread.h>


typedef struct my_item {
  /* TODO: More stuff here, maybe? */
  int args;
  void (*handler)(int);
  struct my_item *next;
  struct my_item *prev;
} my_item_t;

typedef struct my_queue {
  int size;
  my_item_t *head;
  my_item_t *tail;
  /* TODO: More stuff here, maybe? */
} my_queue_t;

void async_init(int);
void async_run(void (*fx)(int), int args);

#endif
