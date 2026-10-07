import atexit
import collections
import itertools
import subprocess
import threading
import signal
import sys
import json
import time
import os

try:
    from transformers import AutoTokenizer
    from rich.live import Live
    from rich.console import Group
    from rich.text import Text
except ImportError:
    print("Please install required libraries: pip install transformers rich")
    sys.exit(1)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.environ.get(
    "MODEL_DIR", os.path.join(REPO_ROOT, "models", "Llama-3.2-1B-Instruct")
)

# A batch is abandoned if the engine goes this long without saying anything about any of its
# requests. Decode emits a token per sequence per step (~130 ms per step on the GTX 1650), so
# silence this long means something is wrong rather than slow, and waiting forever on a
# message that is never coming is the one failure mode that looks like a freeze.
STALL_TIMEOUT_S = float(os.environ.get("STALL_TIMEOUT_S", "120"))

# Prompts only ever come from prompts.txt, one per line: this is a benchmark harness, not a
# chat. The single optional argument is how many of them to run, counted from the top, so
# sweeping batch sizes over the same prompts is just `client.py 1`, `client.py 8`, ...
USAGE = f"usage: {os.path.basename(sys.argv[0])} [number of prompts to run]"
PROMPTS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts.txt")

# MAX_NEW_TOKENS_GENERATED in config.h; the per-request budget this harness asks for.
MAX_NEW_TOKENS = 512

# Argument and file are both checked before anything expensive happens, so a typo fails now
# rather than after the tokenizer and 2.30 GiB of weights have been loaded.
if len(sys.argv) > 2:
    print(USAGE, file=sys.stderr)
    sys.exit(2)
limit = None
if len(sys.argv) == 2:
    try:
        limit = int(sys.argv[1])
    except ValueError:
        print(f"{USAGE}\n'{sys.argv[1]}' is not an integer.", file=sys.stderr)
        sys.exit(2)
    if limit < 1:
        print(f"{USAGE}\nAsked for {limit} prompts; it takes at least 1.", file=sys.stderr)
        sys.exit(2)

try:
    with open(PROMPTS_FILE, "r", encoding="utf-8") as f:
        prompts = [line.strip() for line in f if line.strip()]
except OSError as exc:
    print(f"Could not read prompts from {PROMPTS_FILE}: {exc}", file=sys.stderr)
    sys.exit(1)
if not prompts:
    print(f"No prompts in {PROMPTS_FILE}.", file=sys.stderr)
    sys.exit(1)

available = len(prompts)
if limit is not None:
    if limit > available:
        # Running fewer prompts than asked would quietly report a batch size that is not the
        # one on the command line, which is exactly the number a benchmark gets compared on.
        print(f"Asked for {limit} prompts but {PROMPTS_FILE} only has {available}.",
              file=sys.stderr)
        sys.exit(1)
    prompts = prompts[:limit]
print(f"Running {len(prompts)} of {available} prompt(s) from {PROMPTS_FILE}")

print(f"Loading tokenizer from {MODEL_DIR}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

engine_path = "./build/tiny-vllm"
if not os.path.exists(engine_path):
    print(f"Engine not found at {engine_path}. Please build it first.")
    sys.exit(1)

print("Starting C++ engine... (loading weights takes a while, progress goes to stderr)")
# We redirect stdout to PIPE to read JSON outputs
# We let stderr flow to the console directly so the user can see prefill/decode speeds
engine = subprocess.Popen(
    [engine_path, MODEL_DIR],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=sys.stderr,
    text=True,
    bufsize=1
)

_shutdown_lock = threading.Lock()
_engine_stopped = False


def shutdown_engine():
    """Close stdin and make sure the C++ engine (and its GPU memory) is released.

    The engine loops forever by design, so it will never exit on its own: we have to
    signal it explicitly, otherwise it keeps the CUDA context and KV cache alive after
    the client is gone.
    """
    global _engine_stopped
    with _shutdown_lock:
        if _engine_stopped:
            return
        _engine_stopped = True

    if engine.poll() is None:
        # EOF on stdin first, so the engine's input thread can unwind cleanly
        try:
            if engine.stdin and not engine.stdin.closed:
                engine.stdin.close()
        except (BrokenPipeError, OSError):
            pass

        engine.terminate()
        try:
            engine.wait(timeout=5)
        except subprocess.TimeoutExpired:
            engine.kill()
            try:
                engine.wait(timeout=5)
            except subprocess.TimeoutExpired:
                print(
                    f"Warning: engine (pid {engine.pid}) would not die, kill it manually.",
                    file=sys.stderr,
                )

    try:
        if engine.stdout and not engine.stdout.closed:
            engine.stdout.close()
    except OSError:
        pass


atexit.register(shutdown_engine)


def _handle_signal(signum, _frame):
    shutdown_engine()
    sys.exit(128 + signum)


for _sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
    signal.signal(_sig, _handle_signal)

# Per-request progress state. We no longer stream token text: the display is one line per
# prompt showing which phase it is in (queued -> decode -> done).
#
# "queued" covers everything before the engine says anything about a request: sitting in the
# engine's input queue, and being prefilled. There is no message in between to split those
# two apart - the first thing we hear is the prompt's first token.
STATE_QUEUED = "queued"
STATE_DECODE = "decode"
STATE_DONE = "done"
STATE_FAILED = "failed"
STATE_COLORS = {
    STATE_QUEUED: "white",
    STATE_DECODE: "yellow",
    STATE_DONE: "green",
    STATE_FAILED: "red",
}

# Requests are keyed by an id we mint here, never by the engine's batch slot: a slot is
# recycled as soon as its sequence finishes, so two different prompts can report under the
# same slot within one batch. Keying on the slot used to mean the second completion was
# discarded as a duplicate and the batch never finished.
_next_request_id = itertools.count()
_state_lock = threading.Lock()
requests = {}
_progress = threading.Event()

engine_batch_size = None
engine_ready = threading.Event()


def new_request(prompt):
    """Register a request before it is sent, so no reply can arrive for an unknown id."""
    request_id = next(_next_request_id)
    entry = {
        "id": request_id,
        "prompt": prompt,
        "state": STATE_QUEUED,
        "tokens": 0,
        "done": False,
        "error": None,
        "prefill": {},
        "decode": {},
        "last_update": time.monotonic(),
    }
    with _state_lock:
        requests[request_id] = entry
    return entry


def read_engine_output():
    global engine_batch_size
    for line in iter(engine.stdout.readline, ''):
        if not line:
            break
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            # If the engine prints non-JSON to stdout, ignore it
            continue

        if data.get("type") == "engine_config":
            engine_batch_size = data["batch_size"]
            engine_ready.set()
            continue

        with _state_lock:
            entry = requests.get(data.get("id"))
            if entry is None:
                # a reply for a request we already gave up on, or from an older run
                continue
            entry["last_update"] = time.monotonic()

            if data.get("type") == "preempted":
                # The engine took this sequence's KV pages back and re-queued it; it will
                # be prefilled again and carry on. Not a token and not a completion, so
                # the bare-{id, slot} branch below must not see it.
                entry["state"] = STATE_QUEUED
            elif data.get("type") == "prefill_stats":
                entry["prefill"] = data
                if entry["state"] == STATE_QUEUED:
                    entry["state"] = STATE_DECODE
            elif data.get("done"):
                if not entry["done"]:
                    entry["done"] = True
                    entry["decode"] = data
                    entry["error"] = data.get("error")
                    entry["state"] = STATE_FAILED if entry["error"] else STATE_DONE
            else:
                # A bare {id, slot} line, one per generated token - the engine sends no text.
                # It emits a prompt's first token just before its prefill_stats, so a token
                # is what actually tells us a request has left the queue.
                entry["tokens"] += 1
                if entry["state"] == STATE_QUEUED:
                    entry["state"] = STATE_DECODE
        _progress.set()


output_thread = threading.Thread(target=read_engine_output, daemon=True)
output_thread.start()

if not engine_ready.wait(timeout=300):
    print("Engine never reported its configuration; is it still loading weights?", file=sys.stderr)
    shutdown_engine()
    sys.exit(1)
BATCH_SIZE_CPP = engine_batch_size  # derived from the KV cache in main.cu
print(f"Engine ready: {BATCH_SIZE_CPP} decode slots")

# ---- continuous batching ----
# Everything goes to the engine at once and the engine admits each prompt into whatever
# slot is free at the time (see the admission loop in main.cu). Chunking into groups of
# BATCH_SIZE_CPP used to block here until the slowest sequence of a chunk finished, so
# the tail of every chunk ran with most slots idle - 31 prompts took 63 s even though
# most of them were done long before. Feeding the queue instead keeps all slots busy
# until there is genuinely no work left.
#
# Submitting from its own thread matters: tokenising is not free, and a big enough batch
# fills the 64 KiB stdin pipe and blocks the writer. Neither should stall the display or
# the liveness check.
batch = []
submit_done = threading.Event()
submit_error = None

def submit_all():
    global submit_error
    try:
        for p in prompts:
            messages = [
                {"role": "system", "content": "You are a helpful and detailed AI assistant."},
                {"role": "user", "content": p}
            ]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            tokens = tokenizer.encode(text, add_special_tokens=False)

            entry = new_request(p)
            with _state_lock:
                batch.append(entry)
            # one line per request: "<id> <max_tokens> <token> <token> ..."
            # The output budget is per request on the wire now (see request_queue.h); this
            # harness is the interactive one, so it just asks for the engine's full cap.
            # bench.py is the thing that varies it.
            engine.stdin.write(f"{entry['id']} {MAX_NEW_TOKENS} " + " ".join(map(str, tokens)) + "\n")
            engine.stdin.flush()
            _progress.set()
    except (BrokenPipeError, OSError) as exc:
        submit_error = f"could not submit to the engine: {exc}"
    finally:
        submit_done.set()
        _progress.set()

submit_thread = threading.Thread(target=submit_all, daemon=True)

total_prompts = len(prompts)
idx_width = len(str(total_prompts))
batch_begin = time.perf_counter()
submit_thread.start()

def generate_renderable(height):
    with _state_lock:
        rows = list(enumerate(batch))
        submitted = len(batch)
        counts = collections.Counter(entry["state"] for _, entry in rows)
        live_tokens = sum(entry["tokens"] for _, entry in rows)

    header = Text(
        f"{submitted}/{total_prompts} submitted | {counts[STATE_QUEUED]} queued"
        f" | {counts[STATE_DECODE]} decoding | {counts[STATE_DONE]} done"
        f" | {counts[STATE_FAILED]} failed | {live_tokens} tok"
        f" | {time.perf_counter() - batch_begin:.0f}s",
        style="bold cyan",
    )

    def row_text(index, entry):
        prompt = entry["prompt"]
        preview = prompt if len(prompt) <= 48 else prompt[:45] + "..."
        if entry["error"]:
            preview = f"{entry['error']}: {preview}"
        return Text(
            f"[{index + 1:>{idx_width}}] {entry['state']:<7} {entry['tokens']:>4} tok  {preview}",
            style=STATE_COLORS[entry["state"]],
        )

    # A full batch is usually taller than the terminal, and a Live region that does not
    # fit scrolls into nonsense. Show what is moving and count the rest.
    budget = max(4, height - 3)
    if len(rows) <= budget:
        return Group(header, *(row_text(i, e) for i, e in rows))

    in_flight = [(i, e) for i, e in rows if e["state"] == STATE_DECODE][:budget]
    hidden = len(rows) - len(in_flight)
    return Group(
        header,
        *(row_text(i, e) for i, e in in_flight),
        Text(f"... {hidden} more not shown", style="dim"),
    )

def fail_pending(reason):
    """Give up on whatever the engine still owes us, so the batch can never hang."""
    with _state_lock:
        for entry in batch:
            if not entry["done"]:
                entry["done"] = True
                entry["error"] = reason
                entry["state"] = STATE_FAILED

abandoned = None
with Live(generate_renderable(24), refresh_per_second=10) as live:
    while True:
        with _state_lock:
            pending = sum(1 for entry in batch if not entry["done"])
            last_update = max(
                [entry["last_update"] for entry in batch], default=batch_begin
            )
        # only an empty queue *and* nothing left to submit means we are finished
        if submit_done.is_set() and pending == 0:
            break

        if submit_error is not None:
            abandoned = submit_error
            fail_pending("submit_failed")
            break

        if engine.poll() is not None:
            abandoned = f"engine exited with code {engine.returncode}"
            fail_pending("engine_exited")
            break

        if time.monotonic() - last_update > STALL_TIMEOUT_S:
            abandoned = (
                f"no output from the engine for {STALL_TIMEOUT_S:.0f}s "
                f"with {pending} request(s) unfinished"
            )
            fail_pending("timeout")
            break

        # wait on the reader thread instead of polling a dirty flag, so an update that
        # lands between two ticks cannot be missed
        if _progress.wait(timeout=0.1):
            _progress.clear()
        live.update(generate_renderable(live.console.size.height))
    live.update(generate_renderable(live.console.size.height))
batch_wall = (time.perf_counter() - batch_begin) * 1000.0

print()
if abandoned:
    print(f"Batch abandoned: {abandoned}", file=sys.stderr)
print("-" * 60)
print(f"Performance Statistics ({len(batch)} prompts, {batch_wall:.0f} ms wall):")
total_prefill_tokens = 0
total_decode_tokens = 0
prefill_speeds = []
decode_speeds = []
failed = 0
for i, entry in enumerate(batch):
    ps = entry["prefill"]
    ds = entry["decode"]
    print(f"Prompt {i + 1}:")
    if entry["error"]:
        failed += 1
        print(f"  FAILED: {entry['error']}")
    if ps:
        print(f"  Prefill: {ps.get('speed', 0):.2f} tokens/s ({ps.get('time_ms', 0):.2f} ms,"
              f" {ps.get('prompt_tokens', 0)} prompt tokens)")
        total_prefill_tokens += ps.get("prompt_tokens", 0)
        prefill_speeds.append(ps.get("speed", 0))
    if ds and not entry["error"]:
        print(f"  Decode:  {ds.get('decode_speed', 0):.2f} tokens/s ({ds.get('decode_time', 0):.2f} ms, {ds.get('tokens', 0)} tokens generated)")
        total_decode_tokens += ds.get("tokens", 0)
        decode_speeds.append(ds.get("decode_speed", 0))
if failed:
    print(f"{failed} of {len(batch)} prompt(s) failed")
if prefill_speeds:
    print(f"Avg prefill: {sum(prefill_speeds) / len(prefill_speeds):.2f} tokens/s"
          f" ({total_prefill_tokens} prompt tokens total)")
if decode_speeds:
    print(f"Avg decode per sequence: {sum(decode_speeds) / len(decode_speeds):.2f} tokens/s")
if total_decode_tokens and batch_wall > 0:
    print(f"Aggregate decode throughput: {total_decode_tokens / (batch_wall / 1000.0):.2f} tokens/s"
          f" ({total_decode_tokens} tokens over the whole batch)")
print("-" * 60 + "\n")
sys.stdout.flush()

shutdown_engine()
