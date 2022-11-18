#include "file_system.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

__device__ __managed__ u32 gtime = 0;

/*
  Description:   
  Input:
  Output: 
*/
__device__ void fs_init(FileSystem *fs, uchar *volume, int SUPERBLOCK_SIZE,
							int FCB_SIZE, int FCB_ENTRIES, int VOLUME_SIZE,
							int STORAGE_BLOCK_SIZE, int MAX_FILENAME_SIZE, 
							int MAX_FILE_NUM, int MAX_FILE_SIZE, int FILE_BASE_ADDRESS)
{
  // init variables
  fs->volume = volume;

  // init constants (Unit: byte / B)
  fs->SUPERBLOCK_SIZE = SUPERBLOCK_SIZE;  // 4096
  fs->FCB_SIZE = FCB_SIZE;  //  32
  fs->FCB_ENTRIES = FCB_ENTRIES;    // 1024
  fs->STORAGE_SIZE = VOLUME_SIZE;   // 1085440
  fs->STORAGE_BLOCK_SIZE = STORAGE_BLOCK_SIZE;    //32
  fs->MAX_FILENAME_SIZE = MAX_FILENAME_SIZE;    // 20
  fs->MAX_FILE_NUM = MAX_FILE_NUM;    // 1024
  fs->MAX_FILE_SIZE = MAX_FILE_SIZE;    // 1048576
  fs->FILE_BASE_ADDRESS = FILE_BASE_ADDRESS;    // 36864

  // Extra intermediate variable space (8 bytes)
  fs->MODIFY_TIME = 0;
  fs->CREATE_TIME = 0;

}

/*
  Description:   
  Input:
  Output: 
*/
__device__ void memcpy(uchar *target, uchar *source, int size){
  for(int i = 0; i < size; i++){
    target[i] = source[i];
  }
}

/*
  Description:   
  Input:
  Output: 
*/
__device__ u32 memcmp(uchar *target, uchar *source, int size){
  for(int i = 0; i < size; i++){
    if(target[i] != source[i]) return 1;
  }
  return 0;
}

/*
  Description:   Read the FCB permission bit.
  Input:
  Output: 
*/
__device__ u32 FCB_read_permission(FileSystem *fs, u32 FCB_address){
  uchar *target = &fs->volume[fs->SUPERBLOCK_SIZE + FCB_address*fs->FCB_SIZE + 24];
  return (*target >> 6) & 0b00000011;
}

/*
  Description:   Set the permission bit, 
  Input:
  Output: 
*/
__device__ void FCB_set_permission(FileSystem *fs, u32 FCB_address, u32 option, u32 value){
  uchar *target = &fs->volume[fs->SUPERBLOCK_SIZE + FCB_address*fs->FCB_SIZE + 24];
  if(option) *target = (*target & 0b01111111) + (value << 7); // Option = 1,  set read permission
  else *target = (*target & 0b10111111) + (value << 6); // option = 0,  set write permission
}

/*
  Description:   Read the FCB valid bit. Valid = 1, Invalid = 0.
  Input:
  Output: 
*/
__device__ u32 FCB_read_validbit(FileSystem *fs, u32 FCB_address){
  uchar *target = &fs->volume[fs->SUPERBLOCK_SIZE + FCB_address*fs->FCB_SIZE + 24];
  return (*target >> 5) & 0b00000001;
}

/*
  Description:   Set the FCB permission bit. Valid = 1, Invalid = 0.
  Input:
  Output: 
*/
__device__ void FCB_set_validbit(FileSystem *fs, u32 FCB_address, u32 value){
  uchar *target = &fs->volume[fs->SUPERBLOCK_SIZE + FCB_address*fs->FCB_SIZE + 24];
  fs->volume[fs->SUPERBLOCK_SIZE + FCB_address*fs->FCB_SIZE + 24] = (*target & 0b11011111) + (value << 5);
}

/*
  Description:   Read the FCB filename
  Input:
  Output: 
*/
__device__ void FCB_read_filename(FileSystem *fs, u32 FCB_address, uchar *output){
  uchar *source = &fs->volume[fs->SUPERBLOCK_SIZE + FCB_address*fs->FCB_SIZE];
  memcpy(output, source, 20);
}

/*
  Description:   Set the FCB filename
  Input:
  Output: 
*/
__device__ void FCB_set_filename(FileSystem *fs, u32 FCB_address, uchar *input){
  uchar *target = &fs->volume[fs->SUPERBLOCK_SIZE + FCB_address*fs->FCB_SIZE];
  memcpy(target, input, 20);
}

/*
  Description:   Read the FCB starting point address (Unit: block)
  Input:
  Output: 
*/
__device__ u32 FCB_read_start(FileSystem *fs, u32 FCB_address){
  uchar *source = &fs->volume[fs->SUPERBLOCK_SIZE + FCB_address*fs->FCB_SIZE+20];
  u32 result;
  memcpy((uchar*)&result, source, 2);
  return result;
}

/*
  Description:   Set the FCB starting point address (Unit: block)
  Input:
  Output: 
*/
__device__ void FCB_set_start(FileSystem *fs, u32 FCB_address, u32 value){
  uchar *target = &fs->volume[fs->SUPERBLOCK_SIZE + FCB_address*fs->FCB_SIZE+20];
  memcpy(target, (uchar*)&value, 2);
}

/*
  Description:   Read the FCB starting point address (Unit: block)
  Input:
  Output: 
*/
__device__ u32 FCB_read_size(FileSystem *fs, u32 FCB_address){
  uchar *source = &fs->volume[fs->SUPERBLOCK_SIZE + FCB_address*fs->FCB_SIZE+22];
  u32 result;
  memcpy((uchar*)&result, source, 2);
  return result;
}

/*
  Description:   Set the FCB starting point address (Unit: block)
  Input:
  Output: 
*/
__device__ void FCB_set_size(FileSystem *fs, u32 FCB_address, u32 value){
  uchar *target = &fs->volume[fs->SUPERBLOCK_SIZE + FCB_address*fs->FCB_SIZE+22];
  memcpy(target, (uchar*)&value, 2);
}

/*
  Description:   Read the FCB Last modified time
  Input:
  Output: 
*/
__device__ u32 FCB_read_ltime(FileSystem *fs, u32 FCB_address){
  uchar *source = &fs->volume[fs->SUPERBLOCK_SIZE + FCB_address*fs->FCB_SIZE+26];
  u32 result;
  memcpy((uchar*)&result, source, 3);
  return result;
}

/*
  Description:   Set the FCB Last modified time
  Input:
  Output: 
*/
__device__ void FCB_set_ltime(FileSystem *fs, u32 FCB_address){
  uchar *target = &fs->volume[fs->SUPERBLOCK_SIZE + FCB_address*fs->FCB_SIZE+26];
  memcpy(target, (uchar*)&fs->MODIFY_TIME, 3);
  fs->MODIFY_TIME++;
}

/*
  Description:   Read the FCB created time
  Input:
  Output: 
*/
__device__ u32 FCB_read_ctime(FileSystem *fs, u32 FCB_address){
  uchar *source = &fs->volume[fs->SUPERBLOCK_SIZE + FCB_address*fs->FCB_SIZE+29];
  u32 result;
  memcpy((uchar*)&result, source, 3);
  return result;
}

/*
  Description:   Set the FCB created time
  Input:
  Output: 
*/
__device__ void FCB_set_ctime(FileSystem *fs, u32 FCB_address){
  uchar *target = &fs->volume[fs->SUPERBLOCK_SIZE + FCB_address*fs->FCB_SIZE+29];
  memcpy(target, (uchar*)&fs->MODIFY_TIME, 3);
  fs->CREATE_TIME++;
}

/*
  Description:   Memory Compaction (a very time-comsuming job)
  Input:  N/A
  Output:   N/A
*/
__device__ u32 memory_compaction(FileSystem *fs){
  // TODO
  return 0;
}

/*
  Description:   Check if there is continuous n free blocks. (unit: 32-bytes-large block)
  Input:  n -> requested # of continuous free blocks 
  Output: Positive # -> valid block adddress, -1 -> Need memory compaction, -2 -> Lack of space
*/
__device__ int VCB_Query(FileSystem *fs, u32 n){
  bool found = false;
  int current_cfree_block;
  int total_free_block;
  int result;
  for(int i = 0; i < fs->SUPERBLOCK_SIZE; i++){
    uchar *unit = &fs->volume[i];
    for(u32 j = 0; j < 8; j++){
      uchar bit = (*unit >> (8-i)) & 0b00000001;
      if(bit == 0){
        current_cfree_block++;
        total_free_block++;
        if(current_cfree_block >= n){
          found = true;
          result = i*8 + j - current_cfree_block;
          break;
        }
      }
      else{
        current_cfree_block = 0;
      }
    }
  }
  if(!found){
    if(total_free_block < n) return -2;
    else return -1;
  }
  return result;
}

/*
  Description:   bit-operation on masking, supporting VCB_modification ONLY
  Input:
  Output: 
*/
__device__ void cover(FileSystem *fs, u32 layer, u32 start, u32 end, u32 value){
  u32 mask;
  for(u32 j = start; j < end; j++){
    mask += (1 << (31 - j));
  }
  if(value) fs->volume[layer] |= mask; 
  else fs->volume[layer] &= ~mask; 
}

/*
  Description:   Set the [start, start+size] in VCB to be 0/1.
  Input:
  Output: 
*/
__device__ int VCB_modification(FileSystem *fs, u32 start, u32 size, u32 value){
  u32 start_i = start / 32;
  u32 start_j = start % 32;
  u32 end_i = (start + size) / 32;
  u32 end_j = (start + size) % 32;

  if(start_i == end_i){
    cover(fs, start_i, start_j, end_j, value);
  }
  else{
    cover(fs, start_i, start_j, 32, value);
    cover(fs, end_i, 0, end_j, value);
    for(u32 i = start_i+1; i < end_i; i++){
      fs->volume[i] = (value) ? 0xffffffff : 0x00000000;
    }
  }
}

/*
  Description:   
  Input:
  Output: 
*/
__device__ u32 fs_open(FileSystem *fs, char *s, int op)
{
  // Search the FCB, see if there are 
  uchar file_name[20];
  bool found = false;
  int FCB_address;
  for(FCB_address = 0; FCB_address < fs->FCB_ENTRIES; FCB_address++){
    FCB_read_filename(fs, FCB_address, file_name);
    if(memcmp(file_name,(uchar*)s,20) == 0){
      if(FCB_read_validbit(fs, FCB_address)){
        found = true;
        break;
      }
    }
  }
  if(!found){
    // Initiate a new FCB block with size = 0
    for(FCB_address = 0; FCB_address < fs->FCB_ENTRIES; FCB_address++){
      if(FCB_read_validbit(fs, FCB_address) == 0){
        found = true;
        break;
      }
    }
    FCB_set_filename(fs, FCB_address, (uchar*)s);
    FCB_set_permission(fs, FCB_address, 0, 1);
    FCB_set_permission(fs, FCB_address, 1, 1);
    FCB_set_validbit(fs, FCB_address, 1);
    FCB_set_start(fs, FCB_address, 0);
    FCB_set_size(fs, FCB_address, 1024*1024);   // No actual meaning, not involving memory compaction
    FCB_set_ctime(fs, FCB_address);
    FCB_set_ltime(fs, FCB_address);
  }
  else{
    // Clean up the area.
    if(op == 1){
      u32 start = FCB_read_start(fs, FCB_address);
      u32 size = FCB_read_size(fs, FCB_address);
      VCB_modification(fs, start, size, 0);
    }
  }
  return (FCB_address + (op << 31));
}

/*
  Description:   
  Input:
  Output: 
*/
__device__ void fs_read(FileSystem *fs, uchar *output, u32 size, u32 fp)
{
	/* Implement read operation here */
  u32 FCB_address = fp & 0x7fffffff;
  u32 op = (fp & 0x80000000) >> 31;
  assert(op == 0);
  u32 FCB_block_size = FCB_read_size(fs, FCB_address);
  assert(FCB_block_size >= size);
  /* Read from storage */
  if(size == 0) return;
  u32 FCB_Start = FCB_read_start(fs, FCB_address);
  uchar *source = &fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE + FCB_Start * fs->FCB_SIZE];
  memcpy(output, source, size);
}

/*
  Description:   
  Input:
  Output: 
*/
__device__ u32 fs_write(FileSystem *fs, uchar* input, u32 size, u32 fp)
{
	/* Implement write operation here */
  u32 FCB_address = fp & 0x7fffffff;
  u32 op = (fp & 0x80000000) >> 31;
  assert(op == 1);
  u32 storage_address = VCB_Query(fs, size / fs->FCB_SIZE + 1);
  assert(storage_address > -2);   // Assert there are enough space in total.
  if(storage_address == -1){
    storage_address = memory_compaction(fs);
  }
  uchar *target = &fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE + storage_address * fs->FCB_SIZE];
  memcpy(target, input, size);
  return 0;
}

/*
  Description:   
  Input:
  Output: 
*/
__device__ void fs_gsys(FileSystem *fs, int op)
{
	/* Implement LS_D and LS_S operation here */

  if(op == 0){
    assert(op == 0);
  }
  else if(op == 1){
    assert(op == 1);
  }
}

/*
  Description:   
  Input:
  Output: 
*/
__device__ void fs_gsys(FileSystem *fs, int op, char *s)
{
	/* Implement rm operation here */
  assert(op == 2);
}

// Later to be continued: Time ranking compress