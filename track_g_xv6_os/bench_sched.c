/*
 * User-level benchmark program for xv6 scheduler performance.
 * Spawns 3 concurrent workloads:
 * 1. CPU-intensive: Matrix arithmetic & hashing
 * 2. I/O-intensive: Frequent sleep() yielding simulating console/disk I/O
 * 3. Mixed: Intermittent CPU bursts and I/O
 */

#include "kernel/types.h"
#include "kernel/stat.h"
#include "user/user.h"

void cpu_workload(int id) {
  int i, j;
  volatile long long acc = 0;
  for(i = 0; i < 50000; i++) {
    for(j = 0; j < 500; j++) {
      acc += (i * j) ^ (i + j);
    }
  }
  printf("[CPU Worker %d] Finished acc=%ld\n", id, acc);
  exit(0);
}

void io_workload(int id) {
  int i;
  for(i = 0; i < 40; i++) {
    sleep(1); // yields CPU quantum immediately
  }
  printf("[I/O Worker %d] Completed 40 I/O bursts\n", id);
  exit(0);
}

void mixed_workload(int id) {
  int i, j;
  volatile long long acc = 0;
  for(i = 0; i < 20; i++) {
    for(j = 0; j < 10000; j++) {
      acc += j;
    }
    sleep(1);
  }
  printf("[Mixed Worker %d] Completed acc=%ld\n", id, acc);
  exit(0);
}

int main(int argc, char *argv[]) {
  int i, pid;
  int start_tick = uptime();

  printf("=== xv6 Scheduler Benchmark Starting ===\n");

  // Spawn CPU workers
  for(i = 0; i < 3; i++) {
    pid = fork();
    if(pid == 0) cpu_workload(i);
  }

  // Spawn I/O workers
  for(i = 0; i < 3; i++) {
    pid = fork();
    if(pid == 0) io_workload(i);
  }

  // Spawn Mixed workers
  for(i = 0; i < 2; i++) {
    pid = fork();
    if(pid == 0) mixed_workload(i);
  }

  // Wait for all 8 children
  for(i = 0; i < 8; i++) {
    wait(0);
  }

  int end_tick = uptime();
  printf("=== Benchmark Finished in %d ticks ===\n", end_tick - start_tick);
  exit(0);
}

