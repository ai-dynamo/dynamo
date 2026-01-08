# Memory Profiling Pipeline
This OpenAI Compatible pipeline is profiling memory allocations of dynamo itself in a high-thoughput for large payloads.

files:
```
frontend.py -> HTTP Frontend, implemented in Python with the Rust bindings
backend.py  -> proxy and backend implementation
load_test.py -> Sends http requests to frontend
memory_monitor.py -> Monitors pid memory usage
```

Startup, best run from three different terminals in this order:
```
python ./backend.py &
python ./backend.py --proxy-mode &
python ./frontend.py
```

Then
```
python load_test.py
```
