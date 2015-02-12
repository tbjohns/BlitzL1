#pragma once

#include "common.h"

namespace BlitzL1 {

  class Logger {
    private:
      const char* log_directory;
        unsigned int num_points_logged;
        void log_value(char* name, value_t value);

    public:
      Logger(const char* log_directory) :
        log_directory(log_directory), num_points_logged(0) {}

      void log_point(
        value_t elapsed_time,
        value_t obj
      );
  };

}
