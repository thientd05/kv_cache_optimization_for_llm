#pragma once

// The stdin side of the engine: a reader thread parses prompts off stdin and pushes them
// into a queue the scheduler drains. The queue is the only thing shared with that thread,
// so every touch of it has to hold g_queue_mutex.

#include <deque>
#include <mutex>
#include <vector>

// One prompt as it arrives on stdin. `id` is minted by the client, not by us: a batch slot
// is recycled the moment a sequence finishes, so a slot index cannot identify a request.
// Every message we emit about a request carries this id back, which is what lets the client
// tell "slot 3 finished my 4th prompt" from "slot 3 finished my 29th prompt".
//
// `max_tokens` is the per-request output budget. It exists because a serving benchmark has
// to model the thing that makes KV memory management hard: output length varies per request
// and nobody knows it in advance. A reservation-based cache has to reserve for the worst
// case it is told about; an on-demand cache does not. The reservation policy is the only
// consumer of this number besides the stop condition.
//
// `resumed_tokens` is non-zero only for a sequence that was preempted and re-queued for
// recomputation: `tokens` then holds prompt + everything it had already generated, and
// this says how many of those trailing tokens were generated rather than prompted. The
// scheduler needs the split to keep the output budget and the repetition-penalty mask
// honest across the preemption.
struct Request
{
    int id;
    int max_tokens;
    std::vector<int> tokens;
    int resumed_tokens = 0;
};

// Guards the request queue against the stdin reader thread.
extern std::mutex g_queue_mutex;

// Wire format, one request per line: "<request_id> <max_tokens> <token> <token> ...".
void input_thread_func(std::deque<Request> &queue);
