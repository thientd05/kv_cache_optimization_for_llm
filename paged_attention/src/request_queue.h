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
struct Request
{
    int id;
    std::vector<int> tokens;
};

// Guards the request queue against the stdin reader thread.
extern std::mutex g_queue_mutex;

// Wire format, one request per line: "<request_id> <token> <token> ...".
void input_thread_func(std::deque<Request> &queue);
