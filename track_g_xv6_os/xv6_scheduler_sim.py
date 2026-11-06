"""
Track G: High-Fidelity xv6 OS Scheduler & Buffer Cache Simulator
Simulates xv6 kernel process table, spinlocks, state transitions,
and compares Vanilla Round-Robin vs Multi-Level Feedback Queue (MLFQ).
"""

import numpy as np

class Process:
    """Represents an xv6 kernel struct proc."""
    def __init__(self, pid, ptype, burst_times, arrival_tick=0):
        self.pid = pid
        self.ptype = ptype          # 'CPU', 'IO', 'MIXED'
        self.burst_times = list(burst_times) # alternating [cpu_burst, io_burst, cpu_burst, ...]
        self.current_burst_idx = 0
        self.remaining_cpu = burst_times[0] if burst_times else 0
        self.remaining_io = 0
        self.state = "RUNNABLE"     # RUNNABLE, RUNNING, SLEEPING, ZOMBIE
        self.arrival_tick = arrival_tick
        self.first_run_tick = None
        self.finish_tick = None
        self.total_wait_time = 0
        self.ticks_in_slice = 0
        self.priority = 0           # 0=highest, 1=medium, 2=lowest
        self.total_cpu_executed = 0

class XV6SchedulerSimulator:
    def __init__(self, scheduler_type="ROUND_ROBIN", n_proc=8, aging_ticks=100, seed=42):
        self.scheduler_type = scheduler_type
        self.aging_ticks = aging_ticks
        self.rng = np.random.default_rng(seed)
        self.clock = 0
        self.lock_contentions = 0
        self.context_switches = 0
        self.procs = []
        self._init_workload(n_proc)

    def _init_workload(self, n_proc):
        """Generates 3 standard benchmark workloads."""
        # 3 CPU-bound (bursts of 30 ticks)
        for i in range(3):
            self.procs.append(Process(pid=i+1, ptype="CPU", burst_times=[35, 1, 35, 1, 30], arrival_tick=0))
        # 3 I/O-bound (bursts of 2 ticks CPU, 5 ticks I/O)
        for i in range(3, 6):
            self.procs.append(Process(pid=i+1, ptype="IO", burst_times=[2, 6, 2, 6, 2, 6, 2, 6], arrival_tick=0))
        # 2 Mixed
        for i in range(6, 8):
            self.procs.append(Process(pid=i+1, ptype="MIXED", burst_times=[10, 4, 15, 3, 10], arrival_tick=0))

    def run(self, max_ticks=2000):
        running_proc = None

        while self.clock < max_ticks:
            # Check if all completed
            if all(p.state == "ZOMBIE" for p in self.procs):
                break

            # 1. Update I/O sleeping processes
            for p in self.procs:
                if p.state == "SLEEPING":
                    p.remaining_io -= 1
                    if p.remaining_io <= 0:
                        p.current_burst_idx += 1
                        if p.current_burst_idx < len(p.burst_times):
                            p.remaining_cpu = p.burst_times[p.current_burst_idx]
                            p.state = "RUNNABLE"
                            p.ticks_in_slice = 0
                        else:
                            p.state = "ZOMBIE"
                            p.finish_tick = self.clock

            # 2. Priority aging for MLFQ (boost all RUNNABLE to queue 0 every 100 ticks)
            if self.scheduler_type == "MLFQ" and self.clock % self.aging_ticks == 0:
                for p in self.procs:
                    if p.state in ("RUNNABLE", "RUNNING"):
                        p.priority = 0

            # 3. Scheduler selection logic
            if running_proc is None or running_proc.state != "RUNNING":
                running_proc = self._pick_next_process()
                if running_proc is not None:
                    running_proc.state = "RUNNING"
                    self.context_switches += 1
                    if running_proc.first_run_tick is None:
                        running_proc.first_run_tick = self.clock

            # 4. Advance execution on running process
            if running_proc is not None:
                running_proc.remaining_cpu -= 1
                running_proc.ticks_in_slice += 1
                running_proc.total_cpu_executed += 1

                quantum = (1 << running_proc.priority) if self.scheduler_type == "MLFQ" else 1

                # Check burst completion
                if running_proc.remaining_cpu <= 0:
                    running_proc.current_burst_idx += 1
                    if running_proc.current_burst_idx < len(running_proc.burst_times):
                        # Switch to I/O burst
                        running_proc.remaining_io = running_proc.burst_times[running_proc.current_burst_idx]
                        running_proc.state = "SLEEPING"
                    else:
                        running_proc.state = "ZOMBIE"
                        running_proc.finish_tick = self.clock + 1
                    running_proc = None
                # Check quantum expiration
                elif running_proc.ticks_in_slice >= quantum:
                    running_proc.state = "RUNNABLE"
                    if self.scheduler_type == "MLFQ" and running_proc.priority < 2:
                        running_proc.priority += 1  # Demote priority level
                    running_proc.ticks_in_slice = 0
                    running_proc = None

            # 5. Accumulate wait times for RUNNABLE processes
            for p in self.procs:
                if p.state == "RUNNABLE":
                    p.total_wait_time += 1

            self.clock += 1

        return self._compute_metrics()

    def _pick_next_process(self):
        runnables = [p for p in self.procs if p.state == "RUNNABLE"]
        if not runnables:
            return None

        if self.scheduler_type == "ROUND_ROBIN":
            # Linear scan simulating xv6 proc[] table lock acquisition
            self.lock_contentions += len(self.procs) - len(runnables)
            return runnables[0]

        elif self.scheduler_type == "MLFQ":
            # Priority queues: find highest priority available (lowest queue number)
            for q in range(3):
                q_procs = [p for p in runnables if p.priority == q]
                if q_procs:
                    return q_procs[0]
            return runnables[0]

    def _compute_metrics(self):
        turnarounds = [p.finish_tick - p.arrival_tick for p in self.procs if p.finish_tick is not None]
        wait_times = [p.total_wait_time for p in self.procs]
        response_times = [p.first_run_tick - p.arrival_tick for p in self.procs if p.first_run_tick is not None]

        # Jain's Fairness Index
        cpu_shares = np.array([p.total_cpu_executed for p in self.procs], dtype=float)
        jain_fairness = float((np.sum(cpu_shares) ** 2) / (len(cpu_shares) * np.sum(cpu_shares ** 2)))

        return {
            "Total_Ticks": self.clock,
            "Avg_Turnaround": float(np.mean(turnarounds)),
            "Avg_Wait_Time": float(np.mean(wait_times)),
            "Avg_Response_Time": float(np.mean(response_times)),
            "Context_Switches": self.context_switches,
            "Lock_Contentions": self.lock_contentions,
            "Jains_Fairness_Index": jain_fairness
        }
