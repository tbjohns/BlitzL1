#pragma once

#include "common.h"
#include <vector>


namespace BlitzL1 {
  
  class Loss {
    public:
      const value_t L;
      Loss() : L(1.0) {} 

      void compute_Ax(
        std::vector<value_t> &Ax,
        const value_t *x,
        value_t intercept,
        const Dataset *data) const;

      virtual value_t dual_obj(
        const std::vector<value_t> &theta,
        const Dataset *data,
        value_t theta_scaler = 1.0) const = 0;

      virtual value_t primal_loss(
        const std::vector<value_t> &theta,
        const std::vector<value_t> &aux_dual,
        const Dataset *data) const = 0;

      virtual void compute_dual_points(
        std::vector<value_t> &Ax,
        std::vector<value_t> &theta,
        std::vector<value_t> &aux_dual,
        const Dataset *data) const = 0;

      virtual void compute_H(
        std::vector<value_t> &H,
        const std::vector<value_t> &theta,
        const std::vector<value_t> &Ax,
        const Dataset *data) const = 0;
    
      virtual void apply_intercept_update(
        value_t delta,
        std::vector<value_t> &Ax,
        std::vector<value_t> &theta,
        std::vector<value_t> &aux_dual,
        const Dataset *data) const = 0;
  };

  class LogisticLoss : public Loss {
    public:
      LogisticLoss();

      const value_t L;

      value_t dual_obj(
        const std::vector<value_t> &theta,
        const Dataset *data,
        value_t theta_scaler = 1.0) const;

      value_t primal_loss(
        const std::vector<value_t> &theta,
        const std::vector<value_t> &aux_dual,
        const Dataset* data) const;

      void compute_dual_points(
        std::vector<value_t> &Ax,
        std::vector<value_t> &theta,
        std::vector<value_t> &aux_dual,
        const Dataset *data) const;

      void compute_H(
        std::vector<value_t> &H,
        const std::vector<value_t> &theta,
        const std::vector<value_t> &Ax,
        const Dataset *data) const;
        
      void apply_intercept_update(
        value_t delta,
        std::vector<value_t> &Ax,
        std::vector<value_t> &theta,
        std::vector<value_t> &aux_dual,
        const Dataset *data) const;
  };

  class SquaredLoss : public Loss {
    public:
      SquaredLoss();

      const value_t L;

      value_t dual_obj(
        const std::vector<value_t> &theta,
        const Dataset *data,
        value_t theta_scaler = 1.0) const;

      value_t primal_loss(
        const std::vector<value_t> &theta,
        const std::vector<value_t> &aux_dual,
        const Dataset* data) const;

      void compute_dual_points(
        std::vector<value_t> &Ax,
        std::vector<value_t> &theta,
        std::vector<value_t> &aux_dual,
        const Dataset *data) const;

      void compute_H(
        std::vector<value_t> &H,
        const std::vector<value_t> &theta,
        const std::vector<value_t> &Ax,
        const Dataset *data) const;

      void apply_intercept_update(
        value_t delta,
        std::vector<value_t> &Ax,
        std::vector<value_t> &theta,
        std::vector<value_t> &aux_dual,
        const Dataset *data) const;
  };

}
