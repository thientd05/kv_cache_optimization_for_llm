import atexit
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
    from rich.panel import Panel
except ImportError:
    print("Please install required libraries: pip install transformers rich")
    sys.exit(1)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.environ.get(
    "MODEL_DIR", os.path.join(REPO_ROOT, "models", "Llama-3.2-1B-Instruct")
)

print(f"Loading tokenizer from {MODEL_DIR}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

engine_path = "./build/tiny-vllm"
if not os.path.exists(engine_path):
    print(f"Engine not found at {engine_path}. Please build it first.")
    sys.exit(1)

print("Starting C++ engine...")
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

is_done = []
responses = []
prefill_stats = []
decode_stats = []
is_dirty = False
active_slots = 0

def read_engine_output():
    global responses, is_dirty, active_slots, is_done, prefill_stats, decode_stats
    for line in iter(engine.stdout.readline, ''):
        if not line: break
        try:
            data = json.loads(line)
            slot = data.get("slot", -1)
            
            if data.get("type") == "prefill_stats":
                if slot >= 0 and slot < len(prefill_stats):
                    prefill_stats[slot] = data
                continue

            if slot >= 0 and slot < len(responses):
                if data.get("done"):
                    if not is_done[slot]:
                        is_done[slot] = True
                        active_slots -= 1
                        decode_stats[slot] = data
                        is_dirty = True
                else:
                    responses[slot] += data.get("token", "")
                    is_dirty = True
        except json.JSONDecodeError:
            # If the engine prints non-JSON to stdout, ignore it
            pass

output_thread = threading.Thread(target=read_engine_output, daemon=True)
output_thread.start()

BATCH_SIZE_CPP = 16 # Configured in main.cpp

def clear_screen():
    pass # No longer needed with rich

prompts_from_file = False
file_prompts = []
if len(sys.argv) > 1:
    with open(sys.argv[1], 'r', encoding='utf-8') as f:
        file_prompts = [line.strip() for line in f if line.strip()]
    prompts_from_file = True

while True:
    prompts = []
    if prompts_from_file:
        prompts = file_prompts
    else:
        print("\n" + "="*50)
        print("BATCH PROMPT INPUT")
        print("="*50)
        print("Enter prompts line by line. Leave empty and press Enter to finish batch.")
        
        while True:
            try:
                line = input(f"Prompt {len(prompts) + 1}: ")
                if not line.strip():
                    break
                prompts.append(line.strip())
            except EOFError:
                break
            
    if not prompts:
        break

    # Process prompts in chunks of BATCH_SIZE_CPP to ensure correct slot assignments
    for batch_start in range(0, len(prompts), BATCH_SIZE_CPP):
        batch_prompts = prompts[batch_start:batch_start + BATCH_SIZE_CPP]
        
        responses = [""] * len(batch_prompts)
        is_done = [False] * len(batch_prompts)
        prefill_stats = [{} for _ in range(len(batch_prompts))]
        decode_stats = [{} for _ in range(len(batch_prompts))]
        active_slots = len(batch_prompts)
        is_dirty = True
        
        # Send to engine
        for p in batch_prompts:
            messages = [
                {"role": "system", "content": "You are a helpful and detailed AI assistant."},
                {"role": "user", "content": p}
            ]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            tokens = tokenizer.encode(text, add_special_tokens=False)
            
            token_str = " ".join(map(str, tokens)) + "\n"
            engine.stdin.write(token_str)
            engine.stdin.flush()
            
        # Wait for this batch to finish and render streaming responses
        def generate_renderable():
            panels = []
            for i, p in enumerate(batch_prompts):
                status = "Done" if is_done[i] else "Generating..."
                color = "green" if is_done[i] else "yellow"
                title = f"[Prompt {batch_start + i + 1}] - {status}"
                text = responses[i]
                
                # Prevent terminal overflow glitch in rich.live by truncating live preview
                if len(text) > 300:
                    text = "...\n" + text[-300:]
                    
                panels.append(Panel(text, title=title, border_style=color, subtitle=p))
            return Group(*panels)

        with Live(generate_renderable(), refresh_per_second=15) as live:
            while active_slots > 0:
                if engine.poll() is not None:
                    live.update(generate_renderable())
                    print(
                        f"\nEngine exited unexpectedly with code {engine.returncode}.",
                        file=sys.stderr,
                    )
                    break
                if is_dirty:
                    live.update(generate_renderable())
                    is_dirty = False
                time.sleep(0.05)
            # Final render for batch preview
            live.update(generate_renderable())
            
        # Batch is fully generated, print the complete non-truncated outputs
        print("\n\033[1;32m=== BATCH COMPLETE: FULL RESPONSES ===\033[0m")
        for i, p in enumerate(batch_prompts):
            print(f"\033[1;36m[Prompt {batch_start + i + 1}]:\033[0m {p}")
            print(f"\033[1;37m[Response {batch_start + i + 1}]:\033[0m\n{responses[i]}\n")
            
        print("-" * 50)
        print("Performance Statistics for this batch:")
        for i in range(len(batch_prompts)):
            ps = prefill_stats[i]
            ds = decode_stats[i]
            print(f"Prompt {batch_start + i + 1}:")
            if ps:
                print(f"  Prefill: {ps.get('speed', 0):.2f} tokens/s ({ps.get('time_ms', 0):.2f} ms)")
            if ds:
                print(f"  Decode:  {ds.get('decode_speed', 0):.2f} tokens/s ({ds.get('decode_time', 0):.2f} ms, {ds.get('tokens', 0)} tokens generated)")
        print("-" * 50 + "\n")
        sys.stdout.flush()
        
    print("\nAll prompts in batch completed.")
    if prompts_from_file:
        break

shutdown_engine()
