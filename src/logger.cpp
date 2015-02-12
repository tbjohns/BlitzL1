#include "logger.h"

#include <stdio.h>
#include <string.h>
#include <iostream>

using namespace BlitzL1;

void Logger::log_value(char* name, value_t value) {
  FILE* out_file;
  char file_path[256]; 

  sprintf(file_path, "%s/%s.%d", log_directory, name, num_points_logged);
  out_file = fopen(file_path, "w");
  if (out_file == NULL) {
    std::cerr << "Warning: could not open log path \"" << file_path << "\"" << std::endl;
  } else {
    fprintf(out_file, "%.15f\n", value);
    fclose(out_file);
  }
}

void Logger::log_point(double elapsed_time, value_t obj) {
  if (!strlen(log_directory))
    return;

  char time_str[] = "time";
  log_value(time_str, (value_t) elapsed_time);
  char obj_str[] = "obj";
  log_value(obj_str, obj);

  num_points_logged++;
}

