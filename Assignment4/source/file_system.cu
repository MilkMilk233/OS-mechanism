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
  fs->VALID_BLOCK = 0;
}

/*
  Description:   Copy memory bytes
  Input:
  Output: 
*/
__device__ void memcpy(uchar *target, uchar *source, int size){
  for(int i = 0; i < size; i++){
    target[i] = source[i];
  }
}

/*
  Description:   Compare memory bytes and see if thye're identical
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
  uchar *target = &fs->volume[fs->SUPERBLOCK_SIZE + FCB_address*fs->FCB_SIZE + 25];
  return (*target >> 6) & 0b00000011;
}

/*
  Description:   Set the permission bit, 
  Input:
  Output: 
*/
__device__ void FCB_set_permission(FileSystem *fs, u32 FCB_address, u32 option, u32 value){
  uchar *target = &fs->volume[fs->SUPERBLOCK_SIZE + FCB_address*fs->FCB_SIZE + 25];
  if(option) *target = (*target & 0b01111111) + (value << 7); // Option = 1,  set read permission
  else *target = (*target & 0b10111111) + (value << 6); // option = 0,  set write permission
}

/*
  Description:   Read the FCB valid bit. Valid = 1, Invalid = 0.
  Input:
  Output: 
*/
__device__ u32 FCB_read_validbit(FileSystem *fs, u32 FCB_address){
  uchar *target = &fs->volume[fs->SUPERBLOCK_SIZE + FCB_address*fs->FCB_SIZE + 25];
  return (*target >> 5) & 0b00000001;
}

/*
  Description:   Set the FCB permission bit. Valid = 1, Invalid = 0.
  Input:
  Output: 
*/
__device__ void FCB_set_validbit(FileSystem *fs, u32 FCB_address, u32 value){
  uchar *target = &fs->volume[fs->SUPERBLOCK_SIZE + FCB_address*fs->FCB_SIZE + 25];
  fs->volume[fs->SUPERBLOCK_SIZE + FCB_address*fs->FCB_SIZE + 25] = (*target & 0b11011111) + (value << 5);
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
  memcpy((uchar*)&result, source, 3);
  return result;
}

/*
  Description:   Set the FCB starting point address (Unit: block)
  Input:
  Output: 
*/
__device__ void FCB_set_size(FileSystem *fs, u32 FCB_address, u32 value){
  uchar *target = &fs->volume[fs->SUPERBLOCK_SIZE + FCB_address*fs->FCB_SIZE+22];
  memcpy(target, (uchar*)&value, 3);
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
  Input:  start (Unit:blocks), size (Unit: blocks)
  Output: N/A
*/
__device__ void VCB_modification(FileSystem *fs, u32 start, u32 size, u32 value){
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
  Description:   Memory Compaction (a very time-comsuming job)
  Input:  N/A
  Output:   N/A
*/
__device__ void memory_compaction(FileSystem *fs){
  // TODO
  u32 FCB_address, total_size;
  int last_endpoint = -1;
  int block_size;
  u32 min_start, min_address;
  for(int i = 0; i < fs->VALID_BLOCK; i++){
    min_start = fs->SUPERBLOCK_SIZE*8 + 1;
    for(FCB_address = 0; FCB_address < fs->FCB_ENTRIES; FCB_address++){
      if(!FCB_read_validbit(fs,FCB_address)) continue;
      u32 start = FCB_read_start(fs, FCB_address);
      if(start < min_start && start > last_endpoint){
        min_start = start;
        min_address = FCB_address;
      }
    }
    block_size = (FCB_read_size(fs, min_address) -1) / 8 + 1;
    if(!block_size) continue;
    uchar *dest = &fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE + (last_endpoint+1) * fs->FCB_SIZE];
    uchar *source = &fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE + min_start * fs->FCB_SIZE];
    memcpy(dest, source, block_size*fs->FCB_SIZE);
    FCB_set_start(fs, min_address, last_endpoint+1);
    total_size += block_size;
    last_endpoint += block_size;
  }
  VCB_modification(fs, 0, total_size, 1);
  VCB_modification(fs, total_size, fs->SUPERBLOCK_SIZE*8 - total_size, 0);
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
        // printf("Found\n");
        found = true;
        break;
      }
    }
  }
  if(!found){
    // printf("Not Found\n");
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
    FCB_set_start(fs, FCB_address, pow(2,16) - 1);   // No actual meaning, not involving memory compaction
    FCB_set_size(fs, FCB_address, 0);  
    FCB_set_ctime(fs, FCB_address);
    FCB_set_ltime(fs, FCB_address);
    fs->VALID_BLOCK++;
  }
  else{
    // Clean up the area.
    if(op == G_WRITE){
      u32 start = FCB_read_start(fs, FCB_address);
      int block_size = (FCB_read_size(fs, FCB_address) - 1) / 8 + 1;
      VCB_modification(fs, start, block_size, 0);
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
  // printf("size = %d, FCB_block_size = %d\n",size, FCB_block_size);
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
  int storage_address = VCB_Query(fs, size / fs->FCB_SIZE + 1);
  // printf("storage_address = %d\n",storage_address);
  if(storage_address <= -2) printf("Error! storage_address = %d\n",storage_address);
  assert(storage_address > -2);   // Assert there are enough space in total.
  if(storage_address == -1){
    memory_compaction(fs);
    storage_address = FCB_read_start(fs, FCB_address);
  }
  uchar *target = &fs->volume[fs->SUPERBLOCK_SIZE + fs->FCB_SIZE + storage_address * fs->FCB_SIZE];
  memcpy(target, input, size);
  FCB_set_start(fs, FCB_address, storage_address);
  FCB_set_size(fs, FCB_address, size);
  FCB_set_ltime(fs, FCB_address);
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
  u32 FCB_address, ltime, size, current_max_address;
  u32 last_max_ltime = pow(2,24);
  u32 current_max_ltime = 0;
  u32 last_max_size = fs->MAX_FILE_SIZE + 1;
  u32 current_max_size = 0;
  uchar name[20];

  u32 ctime, current_min_address;
  u32 last_min_ctime = 0;
  u32 current_min_ctime = pow(2,24);

  bool found_inner;
  // printf("Tag 1\n");

  assert(op == LS_D || op == LS_S);
  if(op == LS_D){
    printf("===sort by modified time===\n");
    for(int i = 0; i < fs->VALID_BLOCK; i++){
      current_max_ltime = 0;
      for(FCB_address = 0; FCB_address < fs->FCB_ENTRIES; FCB_address++){
        if(!FCB_read_validbit(fs,FCB_address)) continue;
        ltime = FCB_read_ltime(fs, FCB_address);
        // FCB_read_filename(fs,FCB_address, name);
        // printf("ltime = %d, last_max_ltime = %d, current_max_ltime = %d, current_max_address = %d, current_name = %s\n",ltime,last_max_ltime,current_max_ltime,current_max_address,name);
        if(ltime < last_max_ltime && ltime >= current_max_ltime){
          current_max_ltime = ltime;
          current_max_address = FCB_address;
        }
      }
      // printf("ltime = %d, last_max_ltime = %d, current_max_ltime = %d, current_max_address = %d\n",ltime,last_max_ltime,current_max_ltime,current_max_address);
      FCB_read_filename(fs, current_max_address, name);
      current_max_size = FCB_read_size(fs, current_max_address);
      printf("%s\n",name);
      last_max_ltime = current_max_ltime;
    }
  }
  else{
    printf("===sort by file size===\n");
    // printf("Tag 2, fs->VALID_BLOCK = %d\n",fs->VALID_BLOCK);
    for(int i = 0; i < fs->VALID_BLOCK; i++){
      // printf("Tag 3, i = %d\n",i);
      current_max_size = 0;
      current_min_ctime = pow(2,24);
      found_inner = false;
      for(FCB_address = 0; FCB_address < fs->FCB_ENTRIES; FCB_address++){
        
        if(!FCB_read_validbit(fs,FCB_address)) continue;
        size = FCB_read_size(fs, FCB_address);
        ctime = FCB_read_ctime(fs, FCB_address);
        // FCB_read_filename(fs,FCB_address, name);
        // printf("ltime = %d, size = %d, name = %s, last_max_size = %d, current_max_size = %d\n",ltime, size, name, last_max_size,current_max_size);
        if(size < last_max_size && size > current_max_size){
          FCB_read_filename(fs,FCB_address, name);
          // printf("NORMAL: size = %d, name = %s, current_max_size = %d, last_max_size = %d, ctime = %d\n",size, name,current_max_size,last_max_size,ctime);
          current_max_size = size;
          current_max_address = FCB_address;
          current_min_ctime = ctime;
        }
        else if(size == last_max_size){
          // printf("Bingo!!!!!!!\n");
          // FCB_read_filename(fs,FCB_address, name);
          // printf("CASE1: ctime = %d, size = %d, name = %s, current_max_size = %d, last_max_size = %d\n",ctime, size, name,current_max_size,last_max_size);
          // printf("CASE1 cont'd: last_min_ctime = %d, current_min_ctime = %d\n",last_min_ctime,current_min_ctime);
          // if(current_max_ltime > last_max_ltime) current_max_ltime = 0;
          if(ctime > last_min_ctime){
            if(found_inner == false){
              current_min_ctime = pow(2,24);
              found_inner = true;
            } 
            if(ctime <= current_min_ctime){
              // printf("Change Success: ltime = %d, size = %d, name = %s\n",ltime, size, name);
              current_max_size = size;
              current_max_address = FCB_address;
              current_min_ctime = ctime;
            }
          }
        }
        else if(size == current_max_size){
          // printf("Bingo!!!!!!!\n");
          // FCB_read_filename(fs,FCB_address, name);
          // printf("CASE2: ctime = %d, size = %d, name = %s, current_max_size = %d, last_max_size = %d\n",ctime, size, name,current_max_size,last_max_size);
          // printf("CASE2 cont'd: last_min_ctime = %d, current_min_ctime = %d\n",last_min_ctime,current_min_ctime);
          if(ctime < current_min_ctime){
            // printf("Change Success: ltime = %d, size = %d, name = %s\n",ltime, size, name);
            current_max_size = size;
            current_max_address = FCB_address;
            current_min_ctime = ctime;
          }
        }
      }
      // printf("Tag 4, current_max_size = %d, current_max_address = %d, current_max_ltime = %d\n",current_max_size,current_max_address,current_max_ltime);
      FCB_read_filename(fs, current_max_address, name);
      // printf("Tag 5\n");
      printf("%s %d\n",name, current_max_size);
      last_min_ctime = current_min_ctime;
      last_max_size = current_max_size;
    }
  }
}

/*
  Description:   Implement rm operation here
  Input:
  Output: 
*/
__device__ void fs_gsys(FileSystem *fs, int op, char *s)
{
  assert(op == RM);
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
  if(found){
    FCB_set_validbit(fs, FCB_address, 0);
    u32 start = FCB_read_start(fs, FCB_address);
    int block_size = (FCB_read_size(fs, FCB_address) - 1) / 8 + 1;
    VCB_modification(fs, start, block_size, 0);
    fs->VALID_BLOCK--;
  }
  else{
    printf("Error! The file '%s' does not exists.\n",s);
  }
}

// Later to be continued: Time ranking compress