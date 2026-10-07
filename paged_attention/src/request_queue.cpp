#include "request_queue.h"

#include <iostream>
#include <sstream>
#include <string>

std::mutex g_queue_mutex;

// Wire format, one request per line: "<request_id> <token> <token> ...".
void input_thread_func(std::deque<Request>& queue) {
    std::string line;
    while (std::getline(std::cin, line)) {
        if (line.empty()) continue;
        std::stringstream ss(line);
        int id;
        if (!(ss >> id))
        {
            std::cerr << "Malformed input line (no request id), ignoring: " << line << "\n";
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
        queue.push_back({id, std::move(tokens)});
    }
}
