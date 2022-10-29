#include "virtual_memory.h"
#include <cuda.h>
#include <cuda_runtime.h>

__device__ void init_invert_page_table(VirtualMemory *vm) {

  /*
    Since invert_page_table's space limit is 16KB, so the attribute at most
    (16KB / PAGE_ENTRIES) / sizeof(uint_32) = 4. Here we use 2 attributes.
  */
  for (int i = 0; i < vm->PAGE_ENTRIES; i++) {
    // Valid bit(invalid = 1) | 7* Thread Number | 12* Virtual Address | 12* Physical Address 
    vm->invert_page_table[i] = 0x80000000; 
    vm->invert_page_table[i + vm->PAGE_ENTRIES * 2] = 0xffffffff;    // 17* LRU ranking
  }
}

__device__ void vm_init(VirtualMemory *vm, uchar *buffer, uchar *storage,
                        u32 *invert_page_table, int *pagefault_num_ptr,
                        int PAGESIZE, int INVERT_PAGE_TABLE_SIZE,
                        int PHYSICAL_MEM_SIZE, int STORAGE_SIZE,
                        int PAGE_ENTRIES) {
  // init variables
  vm->buffer = buffer;
  vm->storage = storage;
  vm->invert_page_table = invert_page_table;
  vm->pagefault_num_ptr = pagefault_num_ptr;

  // init constants
  vm->PAGESIZE = PAGESIZE;
  vm->INVERT_PAGE_TABLE_SIZE = INVERT_PAGE_TABLE_SIZE;
  vm->PHYSICAL_MEM_SIZE = PHYSICAL_MEM_SIZE;
  vm->STORAGE_SIZE = STORAGE_SIZE;
  vm->PAGE_ENTRIES = PAGE_ENTRIES;

  // before first vm_write or vm_read
  init_invert_page_table(vm);
}

__device__ uchar vm_read(VirtualMemory *vm, u32 addr) {
  /* Complate vm_read function to read single element from data buffer */

  return 123; //TODO
}

__device__ void vm_write(VirtualMemory *vm, u32 addr, uchar value) {
  /* Complete vm_write function to write value into data buffer */
  bool found = false;
  u32 va_base = addr >> 5;
  u32 offset = addr & 0x0000001f;
  u32 pa, pa_base;
  int empty_pte = -1;
  for(int i = 0; i < PAGE_ENTRIES; i++){
    u32 pte_va_base = (vm->invert_page_table[i] & 0x00FFF000) >> 12;
    u32 pte_valid_bit = vm->invert_page_table[i] & 0x80000000;
    if(pte_va_base == va_base && pte_valid_bit == 0x80000000){
      pa_base = vm->invert_page_table[i] & 0x00000FFF;
      found = true;
      break;
    }
    else if(pte_valid_bit == 0x80000000){
      empty_pte = i;
    }
  }
  // Already in the buffer, so directly write it.
  if(found){
    pa = (pa_base << 5) + offset;
    vm->buffer[pa] = value;
  }
  // Not in buffer
  else{
    // No empty space in buffer, swap by LRU Algorithm
    if(empty_pte == -1){

    }
    // Still some empty space in buffer.
    else{
      pa = (empty_pte << 5) + offset;
      vm->invert_page_table[empty_pte] |= 0x80000000; 
      vm->buffer[pa] = value;
    }
  }

}

__device__ void vm_snapshot(VirtualMemory *vm, uchar *results, int offset,
                            int input_size) {
  /* Complete snapshot function togther with vm_read to load elements from data
   * to result buffer */
}

