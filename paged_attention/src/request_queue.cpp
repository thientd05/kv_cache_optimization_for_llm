#include "request_queue.h"

#include "config.h"

#include <iostream>
#include <sstream>
#include <string>

std::mutex g_queue_mutex;

// Wire format, one request per line:
//     "<request_id> <max_tokens> <n_seqs> <token> <token> ..."
//
// n_seqs is how many answers the client wants for this one prompt, and is 1 for everything
// outside §6.3. It is a required field rather than an optional trailing one because it sits
// before the token ids, which are variable-length; a client that still sends the old
// three-field-free format would have its first token id read as n_seqs, which is why the value
// is range-checked loudly rather than trusted.
void input_thread_func(std::deque<Request>& queue) {
    std::string line;
    while (std::getline(std::cin, line)) {
        if (line.empty()) continue;
        std::stringstream ss(line);
        int id;
        int max_tokens;
        int n_seqs;
        if (!(ss >> id))
        {
            std::cerr << "Malformed input line (no request id), ignoring: " << line << "\n";
            continue;
        }
        if (!(ss >> max_tokens) || max_tokens < 1)
        {
            std::cerr << "Request " << id << " has no usable output budget, ignoring\n";
            continue;
        }
        if (!(ss >> n_seqs) || n_seqs < 1 || n_seqs > MAX_PARALLEL_SEQUENCES)
        {
            std::cerr << "Request " << id << ": n_seqs field is " << n_seqs
                      << ", which is outside 1.." << MAX_PARALLEL_SEQUENCES
                      << ". A client sending the old format without it would land here;"
                      << " ignoring the request rather than guessing.\n";
            continue;
        }
        std::vector<int> tokens;
        int token;
        while (ss >> token) tokens.push_back(token);
        if (tokens.empty())
        {
            std::cerr << "Request " << id << " carries no tokens, ignoring\n";
            continue;
        }
        std::lock_guard<std::mutex> lock(g_queue_mutex);
        for (int seq = 0; seq < n_seqs; ++seq)
        {
            // The last copy may move the token vector; the earlier ones have to copy it.
            queue.push_back({id, max_tokens,
                             seq + 1 == n_seqs ? std::move(tokens) : tokens, 0, seq});
        }
    }
}
