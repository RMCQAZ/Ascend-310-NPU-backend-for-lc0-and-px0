#pragma once
#include <deque>
#include <mutex>
#include <condition_variable>
#include <stdint.h>

template <typename T>
class BlockingQueue
{
public:
    BlockingQueue(uint32_t size) : _size(size) {}

    void push(const T &x)
    {
        std::unique_lock<std::mutex> lock(_mutex);
        while (_queue.size() >= _size)
        {
            _full.wait(lock);
        }
        _queue.push_back(x);
        _empty.notify_one();
    }

    T pop()
    {
        std::unique_lock<std::mutex> lock(_mutex);
        while (_queue.empty())
        {
            _empty.wait(lock);
        }
        const T val = _queue.front();
        _queue.pop_front();
        _full.notify_one();
        return val;
    }

    bool empty()
    {
        std::lock_guard<std::mutex> lock(_mutex);
        return _queue.empty();
    }

private:
    uint32_t _size;
    std::deque<T> _queue;
    std::mutex _mutex;
    std::condition_variable _empty, _full;
};