#pragma once

#include "common.h"
#include <vector>


namespace BlitzL1 {
  
  class Loss {
    protected:
      void compute_Ax(
        const std::vector<value_t> &x,
        value_t intercept,
        Dataset *data,
        std::vector<value_t> &result);
    
    public:
      virtual value_t dual_obj(
        const std::vector<value_t> &theta,
        Dataset *data) = 0;

      virtual value_t primal_loss(
        const std::vector<value_t> &theta,
        const std::vector<value_t> &aux_dual) = 0;

      virtual void compute_dual_points(
        std::vector<value_t> &theta,
        std::vector<value_t> &aux_dual,
        const std::vector<value_t> &x,
        value_t intercept,
        Dataset *data) = 0;

      virtual value_t L() = 0; 

      virtual void compute_H(
        std::vector<value_t> &H,
        const std::vector<value_t> &theta,
        const std::vector<value_t> &aux_dual,
        Dataset *data) = 0;
  };

  class LogisticLoss : public Loss {
    public:
      value_t dual_obj(
        const std::vector<value_t> &theta,
        Dataset *data);

      value_t primal_loss(
        const std::vector<value_t> &theta,
        const std::vector<value_t> &aux_dual);

      void compute_dual_points(
        std::vector<value_t> &theta,
        std::vector<value_t> &aux_dual,
        const std::vector<value_t> &x,
        value_t intercept,
        Dataset *data);

      value_t L(); 

      void compute_H(
        std::vector<value_t> &H,
        const std::vector<value_t> &theta,
        const std::vector<value_t> &aux_dual,
        Dataset *data);
  };

  class SquaredLoss : public Loss {
    public:
      value_t dual_obj(
        const std::vector<value_t> &theta,
        Dataset *data);

      value_t primal_loss(
        const std::vector<value_t> &theta,
        const std::vector<value_t> &aux_dual);

      void compute_dual_points(
        std::vector<value_t> &theta,
        std::vector<value_t> &aux_dual,
        const std::vector<value_t> &x,
        value_t intercept,
        Dataset *data);

      value_t L(); 

      void compute_H(
        std::vector<value_t> &H,
        const std::vector<value_t> &theta,
        const std::vector<value_t> &aux_dual,
        Dataset *data);
  };

}
