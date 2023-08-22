import cProfile
import threading
import simplify_slow

def stop_profiling(profiler):
    print("\nStopping profiling...")
    profiler.disable()
    profiler.dump_stats("output_slow.pstats")
    exit()  # Exit the script after profiling is done

profiler = cProfile.Profile()
profiler.enable()

# Set a timer to stop profiling after, say, 5 seconds
timer = threading.Timer(200.0, stop_profiling, args=[profiler])
timer.start()

simplify_slow.main()

