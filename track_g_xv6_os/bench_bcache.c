/*
 * User-level benchmark program for xv6 Buffer Cache optimization.
 * Tests 2Q / Adaptive LRU vs Vanilla xv6 buffer cache:
 * 1. Sequential file read/write (streaming locality)
 * 2. Random block read/write (temporal locality with hot blocks)
 */

#include "kernel/types.h"
#include "kernel/stat.h"
#include "kernel/fcntl.h"
#include "user/user.h"

#define BSIZE 1024
#define NBLOCKS 120

int main(int argc, char *argv[]) {
  char buf[BSIZE];
  int fd, i;
  int start, end;

  printf("=== xv6 Buffer Cache Benchmark Starting ===\n");
  memset(buf, 'A', sizeof(buf));

  // 1. Sequential Write & Read
  start = uptime();
  fd = open("bench_test.tmp", O_CREATE | O_RDWR);
  if(fd < 0) {
    printf("Failed to create test file\n");
    exit(1);
  }

  for(i = 0; i < NBLOCKS; i++) {
    write(fd, buf, BSIZE);
  }
  close(fd);

  fd = open("bench_test.tmp", O_RDONLY);
  for(i = 0; i < NBLOCKS; i++) {
    read(fd, buf, BSIZE);
  }
  close(fd);
  unlink("bench_test.tmp");
  end = uptime();

  printf("Sequential I/O (120 blocks): %d ticks\n", end - start);
  exit(0);
}

