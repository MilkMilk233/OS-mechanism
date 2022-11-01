#include "virtual_memory.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

__device__ void init_invert_page_table(VirtualMemory *vm) {

  /*
    Since invert_page_table's space limit is 16KB, so the attribute at most
    (16KB / PAGE_ENTRIES) / sizeof(uint_32) = 4. Here we use 2 attributes.
  */
  for (int i = 0; i < vm->PAGE_ENTRIES; i++) {
    // Valid bit(invalid = 1) | 7* Thread Number | 12* Virtual Address | 12* Physical Address 
    vm->invert_page_table[i] = 0x80000000; 
    vm->invert_page_table[i + vm->PAGE_ENTRIES] = 0xffffffff;    // 32* LRU ranking
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

__device__ int page_replacement(VirtualMemory *vm, u32 va_base){
  // 1. Swap out victim page.
  // *vm->pagefault_num_ptr += 1;
  u32 min = 0xffffffff;
  int victim_pn;
  for (int i = 0; i < vm->PAGE_ENTRIES; i++) {
    if(vm->invert_page_table[i + vm->PAGE_ENTRIES] < min){
      min = vm->invert_page_table[i + vm->PAGE_ENTRIES];
      victim_pn = i;
    }
  }
  u32 victim_pa_base = vm->invert_page_table[victim_pn] & 0x00000FFF;
  u32 victim_va_base = (vm->invert_page_table[victim_pn] & 0x00FFF000) >> 12;
  for( int i = 0; i < vm->PAGESIZE; i++){
    vm->storage[(victim_va_base << 5) + i] = vm->buffer[(victim_pa_base << 5) + i];
  }
  // 2. Change PTE to invalid.
  vm->invert_page_table[victim_pn] |= 0x80000000;
  // 3. Swap the desired page in
  for( int i = 0; i < vm->PAGESIZE; i++){
    vm->buffer[(victim_pa_base << 5) + i] = vm->storage[(va_base << 5) + i];
  }
  // 4. Reset the page for new table
  // + Thread ID
  vm->invert_page_table[victim_pn] = (va_base << 12) + victim_pa_base;
  return victim_pn;
}

__device__ uchar vm_read(VirtualMemory *vm, u32 addr) {
  /* Complate vm_read function to read single element from data buffer */
  if(addr > (1 << 17)){
    printf("Error\n");
    return '0';
  } 
  bool found = false;
  u32 va_base = addr >> 5;
  u32 va_offset = addr & 0x0000001f;
  u32 pa, pa_base;
  int empty_pn = -1;
  for(int j = 0; j < vm->PAGE_ENTRIES; j++) vm->invert_page_table[j + vm->PAGE_ENTRIES] -= 1;
  for(int i = 0; i < vm->PAGE_ENTRIES; i++){
    u32 pte_va_base = (vm->invert_page_table[i] & 0x00FFF000) >> 12;
    u32 pte_valid_bit = vm->invert_page_table[i] & 0x80000000;
    if(pte_va_base == va_base && pte_valid_bit == 0){
      pa_base = vm->invert_page_table[i] & 0x00000FFF;
      found = true;
      vm->invert_page_table[i + vm->PAGE_ENTRIES] = 0xffffffff;
      break;
    }
    else if(pte_valid_bit == 0x80000000){
      empty_pn = i;
    }
  }
  // Already in the buffer, so directly write it.
  if(found){
    pa = (pa_base << 5) + va_offset;
    return vm->buffer[pa];
  }
  // Not in buffer
  else{
    *vm->pagefault_num_ptr += 1;
    // No empty space in buffer, swap by LRU Algorithm
    if(empty_pn == -1){
      // Doing place replacement.
      page_replacement(vm, va_base);
      return vm_read(vm, addr);
    }
    // Still some empty space in buffer, rewrite that PTE
    else{
      printf("Error in reading empty blocks!!!\n");
    }
  }
}

__device__ void vm_write(VirtualMemory *vm, u32 addr, uchar value) {
  /* Complete vm_write function to write value into data buffer */
  if(addr > (1 << 17)){
    printf("Error\n");
    return;
  } 
  // printf("Tag1\n");
  bool found = false;
  u32 va_base = addr >> 5;
  u32 va_offset = addr & 0x0000001f;
  u32 pa, pa_base;
  int empty_pn = -1;
  for(int j = 0; j < vm->PAGE_ENTRIES; j++) vm->invert_page_table[j + vm->PAGE_ENTRIES] -= 1;
  for(int i = 0; i < vm->PAGE_ENTRIES; i++){
    u32 pte_va_base = (vm->invert_page_table[i] & 0x00FFF000) >> 12;
    u32 pte_valid_bit = vm->invert_page_table[i] & 0x80000000;
    if(pte_va_base == va_base && pte_valid_bit == 0){
      pa_base = vm->invert_page_table[i] & 0x00000FFF;
      found = true;
      vm->invert_page_table[i + vm->PAGE_ENTRIES] = 0xffffffff;
      break;
    }
    else if(pte_valid_bit == 0x80000000){
      empty_pn = i;
    }
  }
  // Already in the buffer, so directly write it.
  if(found){
    pa = (pa_base << 5) + va_offset;
    vm->buffer[pa] = value;
  }
  // Not in buffer
  else{
    *vm->pagefault_num_ptr += 1;
    // No empty space in buffer, swap by LRU Algorithm
    if(empty_pn == -1){
      // Doing place replacement.
      page_replacement(vm, va_base);
      return vm_write(vm, addr, value);
    }
    // Still some empty space in buffer, rewrite that PTE
    else{
      // Assert it's in the initialization period.
      pa = (empty_pn << 5) + va_offset;
      // + Possible thread ID
      vm->invert_page_table[empty_pn] = (va_base << 12) + empty_pn;
      vm->invert_page_table[empty_pn + vm->PAGE_ENTRIES] = 0xffffffff;
      vm->buffer[pa] = value;
    }
  }
}

__device__ void vm_snapshot(VirtualMemory *vm, uchar *results, int offset,
                            int input_size) {
  /* Complete snapshot function togther with vm_read to load elements from data
   * to result buffer */
  for(int i = 0; i < input_size; i++){
    results[i] = vm_read(vm, i);
  }
}
