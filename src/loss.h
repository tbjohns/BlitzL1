#pragma once

#include "common.h"
#include <vector>


namespace BlitzL1 {
  
  class Loss {
    protected:
      void compute_Ax(
        const value_t *x,
        value_t intercept,
        Dataset *data,
        std::vector<value_t> &result);
    
    public:
      const value_t L;

      virtual value_t dual_obj(
        const std::vector<value_t> &theta,
        Dataset *data) = 0;

      virtual value_t primal_loss(
        const std::vector<value_t> &theta,
        const std::vector<value_t> &aux_dual) = 0;

      virtual void compute_dual_points(
        std::vector<value_t> &theta,
        std::vector<value_t> &aux_dual,
        const value_t *x,
        value_t intercept,
        Dataset *data) = 0;

      virtual void compute_H(
        std::vector<value_t> &H,
        const std::vector<value_t> &theta,
        const std::vector<value_t> &aux_dual,
        Dataset *data) = 0;

      Loss() : L(1.0) {} 
  };

  class LogisticLoss : public Loss {
    public:
      LogisticLoss();

      const value_t L;

      value_t dual_obj(
        const std::vector<value_t> &theta,
        Dataset *data);

      value_t primal_loss(
        const std::vector<value_t> &theta,
        const std::vector<value_t> &aux_dual);

      void compute_dual_points(
        std::vector<value_t> &theta,
        std::vector<value_t> &aux_dual,
        const value_t *x,
        value_t intercept,
        Dataset *data);

      void compute_H(
        std::vector<value_t> &H,
        const std::vector<value_t> &theta,
        const std::vector<value_t> &aux_dual,
        Dataset *data);
  };

  class SquaredLoss : public Loss {
    public:
      SquaredLoss();

      const value_t L;

      value_t dual_obj(
        const std::vector<value_t> &theta,
        Dataset *data);

      value_t primal_loss(
        const std::vector<value_t> &theta,
        const std::vector<value_t> &aux_dual);

      void compute_dual_points(
        std::vector<value_t> &theta,
        std::vector<value_t> &aux_dual,
        const value_t *x,
        value_t intercept,
        Dataset *data);

      void compute_H(
        std::vector<value_t> &H,
        const std::vector<value_t> &theta,
        const std::vector<value_t> &aux_dual,
        Dataset *data);
  };

}
