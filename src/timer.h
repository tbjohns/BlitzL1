#pragma once

#include "common.h"

#include <sys/time.h>

namespace BlitzL1 {

  double get_real_time() {
    timeval time;
    gettimeofday(&time, NULL);

    return (double) (time.tv_sec + (time.tv_usec/1.0e6));
  }

  class Timer {
    private:
      double absolute_start_time;
      double absolute_pause_time;
      double pause_accum;
      bool is_paused;

    public:
      Timer() {
        pause_accum = 0.0;
        absolute_start_time = get_real_time();
        is_paused = false;
      }

      void pause_timing() {
        absolute_pause_time = get_real_time();
        is_paused = true;
      }

      void continue_timing() {
        pause_accum += get_real_time() - absolute_pause_time;
        is_paused = false;
      }

      double elapsed_time() {
        if (is_paused) 
          return absolute_pause_time - absolute_start_time - pause_accum;
        else
          return get_real_time() - absolute_start_time - pause_accum;
      }
  };

}
