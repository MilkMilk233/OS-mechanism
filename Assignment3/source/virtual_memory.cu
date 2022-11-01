#include "virtual_memory.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

__device__ void init_invert_page_table(VirtualMemory *vm) {
  for (u32 i = 0; i < vm->PAGE_ENTRIES; i++) {
    // Valid bit(invalid = 1) | 11* Unused | 4* Thread Number | 16 * Virtual Address
    vm->invert_page_table[i] = 0x80000000; 
    vm->invert_page_table[i + vm->PAGE_ENTRIES] = 0xffffffff;    // 32* LRU ranking
  }
}

__device__ void init_storage_page_table(VirtualMemory *vm) {
  for (u32 i = 0; i < vm->STORAGE_ENTRIES; i++){
    // Valid bit(invalid = 1) | 11* Unused | 4* Thread Number | 16 * Virtual Address
    vm->storage_page_table[i] = 0x80000000; 
  }
}

__device__ void vm_init(VirtualMemory *vm, uchar *buffer, uchar *storage,
                        u32 *storage_page_table, u32 *invert_page_table, 
                        int *pagefault_num_ptr, int PAGESIZE, int INVERT_PAGE_TABLE_SIZE,
                        int PHYSICAL_MEM_SIZE, int STORAGE_SIZE, int PAGE_ENTRIES, 
                        int STORAGE_TABLE_SIZE, int STORAGE_ENTRIES, int THREAD_NUM) {
  // init variables
  vm->buffer = buffer;
  vm->storage = storage;
  vm->storage_page_table = storage_page_table;
  vm->invert_page_table = invert_page_table;
  vm->pagefault_num_ptr = pagefault_num_ptr;

  // init constants
  vm->PAGESIZE = PAGESIZE;
  vm->INVERT_PAGE_TABLE_SIZE = INVERT_PAGE_TABLE_SIZE;
  vm->PHYSICAL_MEM_SIZE = PHYSICAL_MEM_SIZE;
  vm->STORAGE_SIZE = STORAGE_SIZE;
  vm->PAGE_ENTRIES = PAGE_ENTRIES;
  vm->STORAGE_TABLE_SIZE = STORAGE_TABLE_SIZE;
  vm->STORAGE_ENTRIES = STORAGE_ENTRIES;
  vm->THREAD_NUM = THREAD_NUM;

  // before first vm_write or vm_read
  if(threadIdx.x == 0){
    init_invert_page_table(vm);
    init_storage_page_table(vm);
  }
}

__device__ int page_replacement(VirtualMemory *vm, u32 va_base){
  // 1. Find the victim Page Table Entry (PTE)
  u32 min = 0xffffffff;
  u32 victim_pa_base, current_LRU, victim_sa_base, current_ste;
  for (u32 i = 0; i < vm->PAGE_ENTRIES; i++) {
    current_LRU = vm->invert_page_table[i + vm->PAGE_ENTRIES];
    if(current_LRU < min){
      min = current_LRU;
      victim_pa_base = i;
    }
  }
  // 2. Find the victim Storage Table Entry (STE)
  for( int i = 0; i < vm->STORAGE_SIZE; i++){
    current_ste = vm->storage_page_table[i];
    if((current_ste & 0x0000ffff) == va_base && ((current_ste & 0x000f0000) >> 16) == threadIdx.x){
      victim_sa_base = i;
      break;
    }
    else if((current_ste & 0x80000000) == 0x80000000){
      victim_sa_base = i;
      break;
    } 
  }
  // 3. Set PTE and STE as valid.
  vm->storage_page_table[victim_sa_base] = vm->invert_page_table[victim_pa_base];
  vm->invert_page_table[victim_pa_base] = (threadIdx.x << 16) + va_base;
  vm->invert_page_table[victim_pa_base + vm->PAGE_ENTRIES] = 0xffffffff;
  // 4. Exchange the buffer / storage
  for( int i = 0; i < vm->PAGESIZE; i++){
    uchar temp = vm->storage[(victim_sa_base << 5) + i];
    vm->storage[(victim_sa_base << 5) + i] = vm->buffer[(victim_pa_base << 5) + i];
    vm->buffer[(victim_pa_base << 5) + i] = temp;
  }
  return victim_pa_base;
}

__device__ uchar vm_read(VirtualMemory *vm, u32 addr) {
  /* Complate vm_read function to read single element from data buffer */
  if(addr > (1 << 19)){
    printf("Error: Thread numeber > 4\n");
    return '0';
  } 
  u32 va_base = addr >> 5;
  u32 va_offset = addr & 0x0000001f;
  u32 pa, pa_base;
  int empty_pn = -1;
  // LRU decrease by 1
  for(int j = 0; j < vm->PAGE_ENTRIES; j++) vm->invert_page_table[j + vm->PAGE_ENTRIES] -= 1;
  // See if it's inside physical memory
  for(int i = 0; i < vm->PAGE_ENTRIES; i++){
    u32 pte_va_base = (vm->invert_page_table[i] & 0x0000ffff);
    u32 pte_valid_bit = (vm->invert_page_table[i] & 0x80000000) >> 31;
    u32 pte_thread = (vm->invert_page_table[i] & 0x000f0000) >> 16;
    if(pte_va_base == va_base && pte_valid_bit == 0 && pte_thread == threadIdx.x){
      vm->invert_page_table[i + vm->PAGE_ENTRIES] = 0xffffffff;
      pa = (i << 5) + va_offset;
      return vm->buffer[pa];
    }
    else if(pte_valid_bit == 1){
      empty_pn = i;
      vm->invert_page_table[i] &= 0x7fffffff;
      break;
    }
  }
  // Not in buffer
  *vm->pagefault_num_ptr += 1;
  // No empty space in buffer, swap by LRU Algorithm
  if(empty_pn == -1){
    // Doing place replacement.
    pa_base = page_replacement(vm, va_base);
    pa = (pa_base << 5) + va_offset;
    return vm->buffer[pa];
  }
  // Still some empty space in buffer, rewrite that PTE
  else{
    printf("Error in reading empty blocks!!!\n");
  }
}

__device__ void vm_write(VirtualMemory *vm, u32 addr, uchar value) {
  /* Complete vm_write function to write value into data buffer */
  if(addr > (1 << 19)){
    printf("Error: Thread numeber > 4\n");
    return;
  } 
  u32 va_base = addr >> 5;
  u32 va_offset = addr & 0x0000001f;
  u32 pa, pa_base, pte_va_base, pte_valid_bit, pte_thread;
  int empty_pn = -1;
  // LRU decrease by 1
  for(int j = 0; j < vm->PAGE_ENTRIES; j++) vm->invert_page_table[j + vm->PAGE_ENTRIES] -= 1;
  // printf("vm->invert_page_table[128]=%d\n",vm->invert_page_table[128]);
  for(int i = 0; i < vm->PAGE_ENTRIES; i++){
    pte_va_base = (vm->invert_page_table[i] & 0x0000ffff);
    pte_valid_bit = (vm->invert_page_table[i] & 0x80000000) >> 31;
    pte_thread = (vm->invert_page_table[i] & 0x000f0000) >> 16;
    if(pte_va_base == va_base && pte_valid_bit == 0 && pte_thread == threadIdx.x){
      // Founded in physical memory
      vm->invert_page_table[i + vm->PAGE_ENTRIES] = 0xffffffff;
      pa = (i << 5) + va_offset;
      vm->buffer[pa] = value;
      // LRU decrease by 1
      return;
    }
    else if(pte_valid_bit == 1){
      empty_pn = i;
      vm->invert_page_table[i] &= 0x7fffffff;
      break;
    }
  }
  // Not in physical memory, do swapping
  *vm->pagefault_num_ptr += 1;
  // No empty space in buffer, drop one PTE by LRU Algorithm
  if(empty_pn == -1){
    pa_base = page_replacement(vm, va_base);
  }
  // Found empty space in physical memory, ususally happens in initialization.
  else{
    pa_base = empty_pn;
    vm->invert_page_table[empty_pn] = (threadIdx.x << 16) + va_base;
    vm->invert_page_table[empty_pn + vm->PAGE_ENTRIES] = 0xffffffff;
  }
  pa = (pa_base << 5) + va_offset;
  // printf("Write value %d to %d\n",value, pa);
  vm->buffer[pa] = value;
}

__device__ void vm_snapshot(VirtualMemory *vm, uchar *results, int offset,
                            int input_size) {
  /* Complete snapshot function togther with vm_read to load elements from data
   * to result buffer */
  for(int i = 0; i < input_size; i++){
    results[i] = vm_read(vm, i);
  }
}
