# Simple Pipeline Demo

A minimal example of a 3-stage pipeline in Dynamo where each stage transforms a string and passes it to the next stage.

## Architecture

```
Client -> Stage1 -> Stage2 -> Stage3
                              |
       <--------------------<-+
```

Each stage:
1. Receives input from the previous stage (or client)
2. Transforms the string by appending its name
3. Calls the next stage (if not final)
4. Yields the result back

## Usage

### Quick Start

Run all stages and the client with a single script:

```bash
./run.sh
```

### Manual Start

Start each stage in separate terminals, in order (Stage3 first):

```bash
# Terminal 1 - Start Stage 3 (backend)
python3 stage3.py

# Terminal 2 - Start Stage 2 (middle)
python3 stage2.py

# Terminal 3 - Start Stage 1 (entry point)
python3 stage1.py

# Terminal 4 - Run the client
python3 client.py
```

## Expected Output

**Client:**
```
[Client] Connected to pipeline
[Client] Sending: hello
[Client] Received: hello -> stage1 -> stage2 -> stage3_done
[Client] Done
```

**Stage1:**
```
[Stage1] Input: hello
[Stage1] Transformed: hello -> stage1
[Stage1] Output: hello -> stage1 -> stage2 -> stage3_done
```

**Stage2:**
```
[Stage2] Input: hello -> stage1
[Stage2] Transformed: hello -> stage1 -> stage2
[Stage2] Output: hello -> stage1 -> stage2 -> stage3_done
```

**Stage3:**
```
[Stage3] Input: hello -> stage1 -> stage2
[Stage3] Output: hello -> stage1 -> stage2 -> stage3_done
```

## Data Flow

1. Client sends `"hello"` to Stage1
2. Stage1 transforms to `"hello -> stage1"` and calls Stage2
3. Stage2 transforms to `"hello -> stage1 -> stage2"` and calls Stage3
4. Stage3 transforms to `"hello -> stage1 -> stage2 -> stage3_done"` and yields
5. Result flows back through Stage2 -> Stage1 -> Client
