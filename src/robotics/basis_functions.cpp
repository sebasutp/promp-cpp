
#include "robotics/basis_functions.hpp"
#include "robotics/utils.hpp"
#include <vector>

using namespace std;
using namespace arma;

namespace robotics {

  class ScalarPolyBasis::Impl {
    public:
      unsigned int degree;

      Impl() {
        degree = 3;
      }

      Impl(int degree) : degree(degree) {
      }

      Impl(const Impl& b) = default;
      Impl(Impl&& b) = default;
      ~Impl() = default;
      Impl& operator=(const Impl& b) = default;
      Impl& operator=(Impl&& b) = default;

      int dim() const {
        return degree+1;
      }

      vec eval(const double t) const {
        vec ans(degree+1);
        double tmp = 1;
        for (unsigned int i=0; i<=degree; i++) {
          ans[i] = tmp;
          tmp *= t;
        }
        return ans;
      }

      void poly_deriv(vector<double>& v) const {
        for (unsigned int i=0; i<(v.size()-1); i++) {
          v[i] = v[i+1]*(i+1);
        }
        if (v.size() > 1) {
          v.pop_back();
        } else {
          v.back() = 0.0;
        }
      }

      vec deriv(double t, unsigned int order) const {
        vector<double> coefs(degree+1, 1.0);
        for (unsigned int i=0; i<order; i++) {
          poly_deriv(coefs);
        }
        vec ans(degree+1, fill::zeros);
        double tmp = 1;
        for (unsigned int i=0; i<coefs.size(); i++) {
          ans[i+order] = coefs[i]*tmp;
          tmp *= t;
        }
        return ans;
      }

      nlohmann::json to_stream() const {
        nlohmann::json ret;
        nlohmann::json params;
        params["degree"] = degree;
        ret["type"] = "poly";
        ret["params"] = params;
        return ret;
      }
  };

  ScalarPolyBasis::ScalarPolyBasis() : _impl(new ScalarPolyBasis::Impl()) {
  }

  ScalarPolyBasis::ScalarPolyBasis(unsigned int order) {
    _impl = unique_ptr<Impl>(new Impl(order));
  }

  ScalarPolyBasis::ScalarPolyBasis(const ScalarPolyBasis& b) {
    if (b._impl) _impl = unique_ptr<Impl>(new Impl(*b._impl));
  }

  ScalarPolyBasis& ScalarPolyBasis::operator=(const ScalarPolyBasis& b) {
    if (this != &b && b._impl) {
      _impl = unique_ptr<Impl>(new Impl(*b._impl));
    }
    return *this;
  }

  ScalarPolyBasis::ScalarPolyBasis(ScalarPolyBasis&& b) = default;
  ScalarPolyBasis& ScalarPolyBasis::operator=(ScalarPolyBasis&& b) = default;
  ScalarPolyBasis::~ScalarPolyBasis() = default;

  arma::vec ScalarPolyBasis::eval(double t) const {
    return _impl->eval(t);
  }

  arma::vec ScalarPolyBasis::deriv(double t, unsigned int order) const {
    return _impl->deriv(t, order);
  }

  unsigned int ScalarPolyBasis::dim() const {
    return _impl->dim();
  }

  nlohmann::json ScalarPolyBasis::to_stream() const {
    return _impl->to_stream();
  }


  class ScalarGaussBasis::Impl {
    public:
      vec centers;
      double sigma;

      Impl() {
        unsigned int num_basis = 5;
        this->centers = linspace<vec>(0.0,1.0,num_basis);
        this->sigma = 0.25;
      }

      Impl(const vec& centers, double sigma) : centers(centers), sigma(sigma) {
      }

      Impl(const Impl& b) = default;
      Impl(Impl&& b) = default;
      ~Impl() = default;
      Impl& operator=(const Impl& b) = default;
      Impl& operator=(Impl&& b) = default;

      int dim() const {
        return centers.n_elem;
      }

      vec eval(const double t) const {
        vec ans(this->dim());
        for (unsigned int i=0; i<ans.n_elem; i++) {
          double d = t - centers[i];
          ans[i] = exp(-0.5*d*d/(sigma*sigma));
        }
        return ans;
      }

      vec deriv(double t, unsigned int order) const {
        vec v = eval(t);
        vec tmp(v.n_elem);
        double sigma_sq = sigma*sigma;
        for (unsigned int i=0; i<v.n_elem; i++) {
          if (order == 1) {
            tmp[i] = (centers[i]-t)/sigma_sq;
          } else if (order == 2) {
            double d = (t-centers[i])/sigma_sq; 
            tmp[i] = d*d - 1.0/sigma_sq;
          } else {
            throw std::invalid_argument("Derivative for ScalarGaussBasis defined only for orders 1 and 2");
          }
        }
        return v%tmp; //element-wise multiplication
      }

      nlohmann::json to_stream() const {
        nlohmann::json ret;
        nlohmann::json params;
        params["centers"] = vec2json(centers);
        params["sigma"] = sigma;
        ret["type"] = "sqexp";
        ret["params"] = params;
        return ret;
      }
  };

  ScalarGaussBasis::ScalarGaussBasis() : _impl(new Impl()) {
  }

  ScalarGaussBasis::ScalarGaussBasis(const vec& centers, double sigma) {
    _impl = unique_ptr<Impl>(new Impl(centers, sigma));
  }

  ScalarGaussBasis::ScalarGaussBasis(const ScalarGaussBasis& b) {
    if (b._impl) _impl = unique_ptr<Impl>(new Impl(*b._impl));
  }

  ScalarGaussBasis& ScalarGaussBasis::operator=(const ScalarGaussBasis& b) {
    if (this != &b && b._impl) {
      _impl = unique_ptr<Impl>(new Impl(*b._impl));
    }
    return *this;
  }

  ScalarGaussBasis::ScalarGaussBasis(ScalarGaussBasis&& b) = default;
  ScalarGaussBasis& ScalarGaussBasis::operator=(ScalarGaussBasis&& b) = default;
  ScalarGaussBasis::~ScalarGaussBasis() = default;

  arma::vec ScalarGaussBasis::eval(double t) const {
    return _impl->eval(t);
  }

  arma::vec ScalarGaussBasis::deriv(double t, unsigned int order) const {
    return _impl->deriv(t, order);
  }

  unsigned int ScalarGaussBasis::dim() const {
    return _impl->dim();
  }

  nlohmann::json ScalarGaussBasis::to_stream() const {
    return _impl->to_stream();
  }


  class ScalarCombBasis::Impl {
    public:
      vector<shared_ptr<ScalarBasisFun>> basis;
      unsigned int dim;

      void set_basis(const std::vector<std::shared_ptr<ScalarBasisFun>>& basis) {
        this->basis = basis;
        dim = 0;
        for (const auto& x : basis) {
          dim += x->dim();
        }
      }

      vec eval(double t) const {
        vec ans(dim);
        unsigned int start = 0;
        for (const auto& x : basis) {
          unsigned int end = start + x->dim();
          ans(span(start,end-1)) = x->eval(t);
          start = end;
        }
        return ans;
      }

      vec deriv(double t, unsigned int order) const {
        vec ans(dim);
        unsigned int start = 0;
        for (const auto& x : basis) {
          unsigned int end = start + x->dim();
          ans(span(start,end-1)) = x->deriv(t, order);
          start = end;
        }
        return ans;
      }

      nlohmann::json to_stream() const {
        nlohmann::json ret;
        nlohmann::json params;
        vector<nlohmann::json> basis_list;
        for(const auto& x : basis)
            basis_list.push_back(x->to_stream());
        params["basis_list"] = basis_list;
        ret["type"] = "combined";
        ret["params"] = params;
        return ret;
      }
  };

  ScalarCombBasis::ScalarCombBasis(const std::vector<std::shared_ptr<ScalarBasisFun>>& basis) : _impl(new Impl) {
    _impl->set_basis( basis );
  }
  
  ScalarCombBasis::~ScalarCombBasis() = default;

  arma::vec ScalarCombBasis::eval(double t) const {
    return _impl->eval(t);
  }

  arma::vec ScalarCombBasis::deriv(double t, unsigned int order) const {
    return _impl->deriv(t, order);
  }

  unsigned int ScalarCombBasis::dim() const {
    return _impl->dim;
  }

  nlohmann::json ScalarCombBasis::to_stream() const {
    return _impl->to_stream();
  }

};
