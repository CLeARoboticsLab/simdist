import time
from collections import deque


class LoopTimer:
    def __init__(
        self,
        print_interval_sec=1.0,
        window_size=100,
        node=None,
        name="LoopTimer",
        print_info=True,
        warn_rate=None,
    ):
        self.last_time = None
        self.last_print_time = time.time()
        self.print_interval_sec = print_interval_sec
        self.elapsed_times = deque(maxlen=window_size)
        self.node = node
        self.name = name
        self.print_info = print_info
        self.warn_rate = warn_rate
        self._print = True

    def tick(self):
        now = time.time()
        if self.last_time is not None:
            elapsed = now - self.last_time
            self.elapsed_times.append(elapsed)

            if now - self.last_print_time >= self.print_interval_sec and self._print:
                avg = sum(self.elapsed_times) / len(self.elapsed_times)
                if self.print_info:
                    msg = f"[{self.name}] Avg interval: {avg:.6f} s ({1/avg:.2f} Hz)"
                    if self.node and hasattr(self.node, "get_logger"):
                        self.node.get_logger().info(msg)
                    else:
                        print(msg)
                if self.warn_rate is not None:
                    rate = 1 / avg
                    if rate < self.warn_rate:
                        msg = f"[{self.name}] Warning: Rate {rate:.2f} Hz is below threshold {self.warn_rate:.2f} Hz"
                        if self.node and hasattr(self.node, "get_logger"):
                            self.node.get_logger().warn(msg)
                        else:
                            print(msg)
                self.last_print_time = now
        self.last_time = now

    def enable_print(self):
        self._print = True

    def disable_print(self):
        self._print = False

    def _time(self):
        if self.node:
            now = self.node.get_clock().now()
            return now.nanoseconds * 1e-9
        else:
            return time.time()


def test():
    loop_hz = 100.0
    dt = 1.0 / loop_hz
    timer = LoopTimer(print_interval_sec=1.0)

    print(f"Starting simulated {loop_hz:.0f} Hz loop (every {dt*1000:.1f} ms)...")

    try:
        while True:
            loop_start = time.time()

            # Simulated loop body
            timer.tick()

            # Sleep to simulate a fixed loop rate
            elapsed = time.time() - loop_start
            sleep_time = max(0.0, dt - elapsed)
            time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("Stopped.")


if __name__ == "__main__":
    test()
