import statistics
import time
from typing import Any


class Timer:
    """
    A simple timer for measuring execution time of code blocks.
    Supports both 'with' statement and explicit start()/stop() calls.
    Collects statistics (mean, variance, max, min, etc.) over multiple runs.
    Can be enabled or disabled for low overhead when not needed.
    """

    def __init__(self, name: str = "Timer", enabled: bool = True):
        """
        Initializes the Timer.

        Args:
            name (str): A name for this timer instance (useful when having multiple timers).
            enabled (bool): Whether the timer is initially enabled.
        """
        self.name: str = name
        self._enabled: bool = enabled
        self._times: list[float] = []
        self._start_time: float | None = None  # Stores the start time during a 'with' block or after start()

        self._last_acess: int = 0

    @property
    def num_times(self):
        return len(self._times)

    def enable(self) -> None:
        """Enables the timer."""
        self._enabled = True

    def disable(self) -> None:
        """Disables the timer."""
        self._enabled = False

    def is_enabled(self) -> bool:
        """Returns True if the timer is enabled, False otherwise."""
        return self._enabled

    def start(self) -> None:
        """
        Starts the timer.

        Raises:
            RuntimeError: If the timer is already started.
        """
        if not self._enabled:
            return  # Do nothing if disabled

        if self._start_time is not None:
            raise RuntimeError(f"Timer '{self.name}' is already started.")

        self._start_time = time.perf_counter()

    def stop(self) -> None:
        """
        Stops the timer and records the duration.

        Raises:
            RuntimeError: If the timer was not started.
        """
        if not self._enabled:
            return  # Do nothing if disabled

        if self._start_time is None:
            raise RuntimeError(f"Timer '{self.name}' was not started. Call .start() first.")

        end_time = time.perf_counter()
        duration = end_time - self._start_time
        self._times.append(duration)
        self._start_time = None  # Reset start time after stopping

    def __enter__(self) -> "Timer":
        """
        Context management entry point. Starts the timer if enabled.

        Raises:
             RuntimeError: If the timer is already started (via start() or another with).
        """
        if self._enabled:
            if self._start_time is not None:
                raise RuntimeError(f"Timer '{self.name}' is already started. Cannot enter 'with' block.")
            self._start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        """
        Context management exit point. Stops the timer and saves duration if enabled.
        Handles cases where an exception occurred within the with block.
        """
        # Only stop if enabled AND we successfully started within this 'with' block
        if self._enabled and self._start_time is not None:
            end_time = time.perf_counter()
            duration = end_time - self._start_time
            self._times.append(duration)
            self._start_time = None  # Clean up the start time

        # Returning False will propagate any exceptions that occurred within the 'with' block
        return False

    def reset(self) -> None:
        """Clears all collected timing data and stops the timer if it was running."""
        self._times = []
        self._start_time = None  # Ensure timer is stopped

    def get_times(self) -> list[float]:
        """Returns a copy of the list of all collected execution times."""
        # Return a copy to prevent external modification of internal data
        return self._times[:]

    def get_last(self, n=1) -> float | None:
        if not self._times:
            return None
        return statistics.mean(self._times[-n:])

    def acess_last(self) -> float | None:
        if not self._times:
            return None
        v = self._times[self._last_acess :]
        if len(v) == 0:
            return None
        self._last_acess = len(self._times)
        return statistics.mean(v)

    def get_stats(self) -> dict[str, Any] | None:
        """
        Calculates and returns the statistics of the recorded times.

        Returns:
            Optional[Dict[str, Any]]: A dictionary containing statistics (count, total,
            mean, max, min, variance, stdev) or None if no data has been collected.
        """
        if not self._times:
            return None

        count = len(self._times)
        total = sum(self._times)
        mean = statistics.mean(self._times)
        max_time = max(self._times)
        min_time = min(self._times)

        # Variance and standard deviation require at least 2 samples
        variance = None
        stdev = None
        if count >= 2:
            try:
                # statistics.variance and statistics.stdev can raise StatisticsError
                # for n < 2, but we handle that with the count check.
                variance = statistics.variance(self._times)
                stdev = statistics.stdev(self._times)
            except statistics.StatisticsError:
                # This should ideally not happen due to the count check.
                pass

        stats = {
            "name": self.name,
            "count": count,
            "total": total,
            "mean": mean,
            "max": max_time,
            "min": min_time,
            "variance": variance,
            "stdev": stdev,
        }
        return stats

    def report(self) -> None:
        """Prints the calculated statistics to the console."""
        stats = self.get_stats()
        if stats is None:
            print(f"Timer '{self.name}': No data collected yet.")
        else:
            print(f"--- Timer Stats: '{stats['name']}' ---")
            print(f"  Count:    {stats['count']}")
            print(f"  Total:    {stats['total']:.6f} seconds")
            print(f"  Mean:     {stats['mean']:.6f} seconds")
            print(f"  Max:      {stats['max']:.6f} seconds")
            print(f"  Min:      {stats['min']:.6f} seconds")
            if stats["stdev"] is not None:
                print(f"  Stdev:    {stats['stdev']:.6f} seconds")
            if stats["variance"] is not None:
                print(f"  Variance: {stats['variance']:.6f} seconds")
            print("------------------------------------")


# --- Usage Examples ---

if __name__ == "__main__":
    import random

    # Create timer instances
    process_timer = Timer(name="Process Step")
    io_timer = Timer(name="IO Operations", enabled=True)  # Initially enabled

    print("--- Running simulation with mixed usage ---")

    # Use with statement for some parts
    print("\n--- Using 'with' statements ---")
    for i in range(3):
        print(f"  Iteration {i + 1} (with)")
        with process_timer:
            time.sleep(random.uniform(0.05, 0.15))  # Simulate processing

        with io_timer:
            time.sleep(random.uniform(0.01, 0.03))  # Simulate IO

    # Use start/stop for other parts
    print("\n--- Using start()/stop() methods ---")
    for i in range(3):
        print(f"  Iteration {i + 1} (start/stop)")
        process_timer.start()
        time.sleep(random.uniform(0.05, 0.15))  # Simulate processing
        process_timer.stop()

        # Example of disabling/enabling and using start/stop
        if i == 1:
            print("  Disabling IO timer temporarily...")
            io_timer.disable()
            io_timer.start()  # This call will do nothing because timer is disabled
            time.sleep(0.05)
            io_timer.stop()  # This call will do nothing because timer is disabled
            print("  Re-enabling IO timer...")
            io_timer.enable()
        else:
            io_timer.start()
            time.sleep(random.uniform(0.01, 0.03))  # Simulate IO
            io_timer.stop()

    print("\n--- Reporting Final Stats ---")
    process_timer.report()
    io_timer.report()

    # Example of incorrect usage (will raise RuntimeError)
    print("\n--- Demonstrating Error Handling ---")
    try:
        print("Attempting to start already started timer...")
        process_timer.start()
        process_timer.start()  # Should raise RuntimeError
    except RuntimeError as e:
        print(f"Caught expected error: {e}")
    finally:
        # Need to stop the timer that was started before the error for cleanup
        # In real code, ensure proper error handling or avoid such logic
        if process_timer._start_time is not None:
            process_timer.stop()  # Clean up the manually started timer

    try:
        print("\nAttempting to stop timer that wasn't started...")
        # Assuming io_timer is currently stopped
        io_timer.stop()  # Should raise RuntimeError
    except RuntimeError as e:
        print(f"Caught expected error: {e}")

    try:
        print("\nAttempting to enter 'with' block when timer is manually started...")
        process_timer.start()
        with process_timer:  # Should raise RuntimeError
            time.sleep(0.1)
    except RuntimeError as e:
        print(f"Caught expected error: {e}")
    finally:
        # Clean up the manually started timer
        if process_timer._start_time is not None:
            process_timer.stop()

    try:
        print("\nAttempting to manually start timer inside 'with' block...")
        with process_timer:
            print("  Inside with block...")
            process_timer.start()  # Should raise RuntimeError
            time.sleep(0.1)
        process_timer.stop()  # This won't be reached if error is raised
    except RuntimeError as e:
        print(f"Caught expected error: {e}")

    # Note: The finally block associated with the 'with' statement (__exit__)
    # will correctly handle the timer cleanup if an exception occurs *within*
    # the 'with' block itself, assuming the timer was started by the 'with'.
    # However, if the error is raised by timer.start() *inside* the with,
    # the timer wasn't successfully started by timer.start(), and the __exit__
    # for the outer 'with' might clean up if timer was started by __enter__.
    # It's crucial *not* to nest `start/stop` and `with` for the same timer instance
    # for the same measurement period. Use one method or the other per logical block.
