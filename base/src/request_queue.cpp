#include "request_queue.h"

#include <iostream>
#include <sstream>
#include <string>

std::mutex g_queue_mutex;

// Wire format, one request per line: "<request_id> <max_tokens> <token> <token> ...".
void input_thread_func(std::deque<Request>& queue) {
    std::string line;
    while (std::getline(std::cin, line)) {
        if (line.empty()) continue;
        std::stringstream ss(line);
        int id;
        int max_tokens;
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
        std::vector<int> tokens;
        int token;
        while (ss >> token) tokens.push_back(token);
        if (tokens.empty())
        {
            std::cerr << "Request " << id << " carries no tokens, ignoring\n";
            continue;
        }
        std::lock_guard<std::mutex> lock(g_queue_mutex);
        queue.push_back({id, max_tokens, std::move(tokens), 0});
    }
}
