#include <assert.h>
#include <dirent.h>  // Open the folder in /proc
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

typedef struct pidinfo {
  char name[50];
  __pid_t pid;
  __pid_t ppid;
  __pid_t tid;
} PidInfo;
/* Define a struct, containing its pid, ppid, tid. */
PidInfo pidinfos[10000];
int pid_count = 0;
int graph_count = 0;
int graph[50];

void read_ppid_and_name(__pid_t pid, __pid_t sub_pid, int if_show_pid) {
  char str_pid[20];
  sprintf(str_pid, "%d", pid);
  char str_sub_pid[20];
  sprintf(str_sub_pid, "%d", sub_pid);
  char path[30] = "/proc/";
  strcat(path, str_pid);
  strcat(path, "/task/");
  strcat(path, str_sub_pid);
  strcat(path, "/stat");
  FILE *fp = fopen(path, "r");
  if (fp) {
    char name[50];
    char process_status;              // Process Status, seize a seat only
    char useless;
    __pid_t _pid, ppid;  // seize a seat only
    fscanf(fp, "%d (%[^)]s", &_pid, name);  // Only read words between '(' ')'
    fscanf(fp, "%c %c %d", &useless, &process_status, &ppid);
    for(int k = 0; k < strlen(name); k++){
      if(name[k] == '('){
        rewind(fp);
        fscanf(fp, "%d (%s %c %d", &_pid, name, &process_status, &ppid);
        name[strlen(name) - 1] = '\0';
        break;
      }
    }
    // name[strlen(name) - 1] = '\0';
    if(pid != sub_pid){
      name[strlen(name)+2] = '\0';
      name[strlen(name)+1] = '}';
      for(int i = strlen(name); i > 0; i--){
        name[i] = name[i-1];
      }
      name[0] = '{';
    }
    if(if_show_pid){
      strcat(name,"(");
      strcat(name,str_sub_pid);
      strcat(name,")");
    }
    strcpy(pidinfos[pid_count].name, name);
    pidinfos[pid_count].ppid = ppid;
    // printf("name=%s,ppid=%d\n", name, ppid);
    fclose(fp);
    return;
  } else {
    printf("Error: file open failed at '%s'\n", path);
    exit(1);
  }
}

/* Search the /proc file with the help of `opendir()`, `readdir()` and
  `closedir()` Trying to get information about process & thread */
void search_process_info(int if_show_pid) {
  int pid = 0, sub_pid = 0;
  struct dirent *dir_file, *subdir_file;
  char *folder_name;
  DIR *dir, *subdir;  // Store the structure of the folder

  if (!(dir = opendir("/proc"))) {
    printf("Can't open '/proc': Permission denied.\n");
    return;
  }
  while ((dir_file = readdir(dir)) != NULL) {
    /*  Find the hidden thread folder, e.g. ".243" */
    if ((pid = atoi(dir_file->d_name)) == 0) {
      continue;
    } else {  // store in pidinfo (name and pid and ppid)
      /* First look for threads */
      char path[30] = "/proc/";
      strcat(path, dir_file->d_name);
      strcat(path, "/task");
      if (!(subdir = opendir(path))) {
        printf("Can't open '%s': Permission denied.\n", path);
      } else {
        while ((subdir_file = readdir(subdir)) != NULL) {
          if ((sub_pid = atoi(subdir_file->d_name)) == 0) {
            continue;
          } else {
            pidinfos[pid_count].pid = pid;
            pidinfos[pid_count].tid = sub_pid;
            // printf("Read: pid=%d,tpid=%d,", pid, sub_pid);
            read_ppid_and_name(pid, sub_pid, if_show_pid);
            pid_count++;
          }
        }
      }
      closedir(subdir);
    }
  }
  closedir(dir);
  // for(int k = 0; k < pid_count; k++){
  //   printf("Read: pid = %d, tid = %d, ppid = %d, name = %s\n",pidinfos[k].pid, pidinfos[k].tid, pidinfos[k].ppid, pidinfos[k].name );
  // }
  return;
}


// 把这里当作主战场！！！不要把问题留到 main function.
/* 功能：
    1. 打印当前PID信息
    2. 递归激活子进程， 同时画线
*/
void print_tree(int if_compressed, PidInfo *current_pid, int line_distance, int if_duplicated){

  // 寻找ppid为current_pid的所有进程
  PidInfo sub_pidinfos[300];
  int count_subprocess = 0;
  int is_thread = 0;
  if(current_pid->pid == current_pid->tid){
    for(int i = 0; i < pid_count; i++){
      // For process
      if(pidinfos[i].pid == pidinfos[i].tid){
        if(pidinfos[i].ppid == current_pid->pid){
          sub_pidinfos[count_subprocess] = pidinfos[i];
          count_subprocess++;
        }
      }
      // For thread
      else{
        if(pidinfos[i].pid == current_pid->pid){
          sub_pidinfos[count_subprocess] = pidinfos[i];
          count_subprocess++;
        }
      }
    }
    // 对列表里的子进程进行排序，按名字顺序或pid排序，从小到大。
    if(count_subprocess > 1){
      char temp[50];
      __pid_t temp_pid;
      __pid_t temp_ppid;
      __pid_t temp_tid;
      PidInfo temp_process;

      for(int i = 0; i < count_subprocess - 1; i++){
        for(int j = i + 1; j < count_subprocess; j++){
          if(strcmp(sub_pidinfos[i].name, sub_pidinfos[j].name) > 0){
            temp_process = sub_pidinfos[i];
            sub_pidinfos[i] = sub_pidinfos[j];
            sub_pidinfos[j] = temp_process;
          }
        }
      }
    }
  }

  printf("%s",current_pid->name);
  line_distance += strlen(current_pid->name);

  

  // 说明这是一个叶子进程（没有子进程），此时应当输出了。
  if(count_subprocess == 0){
    printf("\n");
    return;
  }
  // 只有一个子进程
  else if(count_subprocess == 1){
    printf("---");
    current_pid = &sub_pidinfos[0];
    print_tree(if_compressed, current_pid, line_distance + 3, if_duplicated);
    return;
  }
  // 有多个子进程
  else{
    // 在光标处加一个分隔符！
    graph[graph_count] = line_distance+1;
    graph_count++; 
    for(int i = 0; i < count_subprocess; i++){
      // 第一个！直接用printf直球打印当前的，不换行
      if(i == 0){
        printf("-+-");
        current_pid = &sub_pidinfos[0];
        print_tree(if_compressed, current_pid, line_distance + 3, if_duplicated);
      }
      // 最后一个！打印graph + 当前的，只不过最后一个"|"变成“ ` ”号。
      else if(i == count_subprocess - 1){
        for(int j = 0; j < graph_count; j++){
          if(j == 0){
            printf("%*s",graph[0],"");
          }
          else{
            printf("%*s",graph[j]-graph[j-1]-1,"");
          }
          if(j == graph_count-1){
            printf("`-");
            graph_count--;
          }
          else{
            printf("|");
          }
        }
        current_pid = &sub_pidinfos[i];
        print_tree(if_compressed, current_pid, line_distance + 3, if_duplicated);
      }
      // 其他！用printf打印graph + 当前的，不换行
      else{
        for(int j = 0; j < graph_count; j++){
          if(j == 0){
            printf("%*s",graph[0],"");
            printf("|");
          }
          else{
            printf("%*s",graph[j]-graph[j-1]-1,"");
            printf("|");
          }
          if(j == graph_count-1){
            printf("-");
          }
        }
        current_pid = &sub_pidinfos[i];
        print_tree(if_compressed, current_pid, line_distance + 3, if_duplicated);
      }
    }
    return;
  }

}

int main(int argc, char *argv[]) {
  if (argc == 1) {
    // If_show_pid = false
    search_process_info(0);
    // if_compressed = true, starting from pid=1.
    print_tree(1, &pidinfos[0], 0, 0);
  } else {
    int o;
    const char *optstring = "Vclap";
    while ((o = getopt(argc, argv, optstring)) != -1) {
      switch (o) {
        case 'V':
          printf(
              "pstree (PSmisc) 22.21 \nCopyright (C) 1993-2009 Werner "
              "Almesberger and Craig Small\nPSmisc comes with ABSOLUTELY NO "
              "WARRANTY.\nThis is free software, and you are welcome to "
              "redistribute it under\nthe terms of the GNU General Public "
              "License.\nFor more information about these matters, see the "
              "files "
              "named COPYING.\n");
          break;
        case 'c':
          // If_show_pid = false
          search_process_info(0);
          // if_compressed = false, starting from pid=1.
          print_tree(0, &pidinfos[0], 0, 0);
          break;
        case 'l':
          // Same as default.
          // If_show_pid = false
          search_process_info(0);
          // if_compressed = true, starting from pid=1.
          print_tree(1, &pidinfos[0], 0, 0);
          break;
        case 'a':
          printf("opt is a, oprarg is: %s\n", optarg);
          break;
        case 'p':
          search_process_info(1);
          print_tree(0, &pidinfos[0], 0, 0);
          break;
        case '?':
          printf("Usage: pstree [ -a ] [ -c ] [ -l ] [ -p ]\n");
          printf("       pstree -V\n");
          printf("Display a tree of processes.\n\n");
          printf(
              "       pstree -a                  show command line "
              "arguments\n");
          printf(
              "       pstree -c                  don't compact identical "
              "subtrees\n");
          printf(
              "       pstree -l                  don't truncate long lines\n");
          printf("       pstree -p                  show PIDs; implies -c\n");
          printf(
              "       pstree -V                  display version "
              "information\n");
          break;
      }
    }
  }
  return 0;
}
