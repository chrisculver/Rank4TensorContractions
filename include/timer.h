#ifndef TIMER_H
#define TIMER_H

#include <chrono>
#include <atomic>
#include <string>
#include <iostream>

template <typename Clock = std::chrono::high_resolution_clock>
class Timer
{
  const typename Clock::time_point start_point;
  const std::string name;
  public:
    Timer(std::string n) : 
      start_point(Clock::now()), name(n)
    {}
        
    template <typename Rep = typename Clock::duration::rep, typename Units = typename Clock::duration>
    Rep elapsed_time() const
    {
      std::atomic_thread_fence(std::memory_order_relaxed);
      auto counted_time = std::chrono::duration_cast<Units>(Clock::now() - start_point).count();
      std::atomic_thread_fence(std::memory_order_relaxed);
      return static_cast<Rep>(counted_time);
    }
    
    template <typename Units = typename Clock::duration>
    void print(std::string unit) const {
      std::cout << "TIMER: " << name << " ELAPSED: " << elapsed_time<unsigned int, Units>() << " " << unit << std::endl;
    }
};

#endif
