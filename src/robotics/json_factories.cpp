
#include <unordered_map>
#include "robotics/json_factories.hpp"
#include "robotics/utils.hpp"
#include <armadillo>
#include <fstream>

using namespace std;
using namespace arma;

namespace robotics {

  /**
   * Creates a ScalarPolyBasis type with parameters given in JSON format. The JSON object
   * must contain a property called "degree" containing an integer number that represents
   * the degree of the polynomial.  
   * @brief Creates a ScalarPolyBasis type with parameters given in JSON format
   * @since version 0.1.1
   */ 
  std::shared_ptr<ScalarBasisFun> json2ScalarPolyBasis(const nlohmann::json& params) {
    unsigned int degree = params.at("degree");
    return shared_ptr<ScalarPolyBasis>( new ScalarPolyBasis(degree) );
  }

  /**
   * Creates a ScalarGaussBasis type with parameters given in JSON format. The JSON object
   * must contain two properties called "centers" and "sigma" representing respectively
   * a vector with the scalar centers of the Radial Basis functions and its radius (See
   * ScalarGaussBasis class documentation).
   * @brief Creates a ScalarGaussBasis type with parameters given in JSON format
   * @since version 0.1.1
   */ 
  std::shared_ptr<ScalarBasisFun> json2ScalarGaussBasis(const nlohmann::json& params) {
    const vec& centers = json2vec(params.at("centers"));
    double sigma = params.at("sigma");
    return shared_ptr<ScalarGaussBasis>( new ScalarGaussBasis(centers, sigma) );
  }

  /**
   * Creates a ScalarCombBasis type with parameters given in JSON format. The JSON object
   * must contain two properties called "params" and "conf" representing respectively
   * a vector with the parameters of the Basis functions and a list with the configuration
   * of each basis function.
   * @brief Creates a ScalarGaussBasis type with parameters given in JSON format
   * @since version 0.1.1
   */ 
  std::shared_ptr<ScalarBasisFun> json2ScalarCombBasis(const nlohmann::json& basis) {
    const vec& params = json2vec(basis.at("params"));
    const auto& conf = basis.at("conf");
    vector<shared_ptr<ScalarBasisFun>> ans;
    unsigned int start = 0;
    for (const auto& sub_basis : conf) {
      unsigned int nparams = sub_basis.at("nparams");
      unsigned int end = start + nparams;
      string type = sub_basis.at("type");
      if (type == "sqexp") {
        double sigma = exp(params[start]);
        vec centers = params(span(start+1, end-1));
        ans.push_back( shared_ptr<ScalarBasisFun>( new ScalarGaussBasis(centers, sigma) ) );
      } else if (type == "poly") {
        unsigned int order = sub_basis.at("conf").at("order");
        ans.push_back( shared_ptr<ScalarBasisFun>( new ScalarPolyBasis(order) ) );
      }
      start = end;
    }
    return shared_ptr<ScalarBasisFun>( new ScalarCombBasis(ans) );
  }

  /**
   * Creates a ProMP object the parameters given in JSON format. The received JSON
   * object is expected to have three properties called "mu_w", "Sigma_w" and "Sigma_y".
   * The value of "mu_w" should be a vector loadable with the json2vec method and both
   * "Sigma_w" and "Sigma_y" should contain matrices loadable with json2mat. For the
   * interpretation of these parameters refer to the ProMP class.
   * @brief Creates a ScalarGaussBasis type with parameters given in JSON format
   * @since version 0.1.1
   */
  ProMP json2basic_promp(const nlohmann::json& params) {
    vec mu_w = json2vec(params.at("mu_w"));
    mat Sigma_w = json2mat(params.at("Sigma_w"));
    mat Sigma_y = json2mat(params.at("Sigma_y"));
    return ProMP(mu_w, Sigma_w, Sigma_y);
  }

  /**
   * Creates a FullProMP object from the parameters given in JSON format. The received JSON
   * object is expected to have three properties named "kernel", "model" and "num_joints". The
   * value of the "kernel" property must be loadable with the factory class ScalarBasisFactory,
   * the property "model" must be loadable with the method json2basic_promp, and num_joints be
   * an unsigned integer with the number of joints that the robot has.
   * @brief Creates a FullProMP class from a JSON object
   */
  FullProMP json2full_promp(const nlohmann::json& params) {
    shared_ptr<ScalarBasisFun> kernel; 
    if (params.count("kernel")) {
      kernel = ScalarBasisFactory::instance().construct(params.at("kernel"));
    } else if (params.count("basis")) {
      kernel = json2ScalarCombBasis(params.at("basis"));
    } else {
      throw std::logic_error("The JSON format for a Full ProMP does not contain a valid "
          "basis function specification");
    }
    ProMP model = json2basic_promp(params.at("model"));
    unsigned int num_joints = params.at("num_joints");
    return FullProMP(kernel, model, num_joints);
  }

  class ScalarBasisFactory::Impl {
    public:
      unordered_map<string, ConstructorType> constructors;

      Impl() {
        constructors["sqexp"] = json2ScalarGaussBasis;
        constructors["poly"] = json2ScalarPolyBasis;
      }

      shared_ptr<ScalarBasisFun> construct(const json& obj) const {
        const string& key = obj.at("type");
        const json& params = obj.at("params");
        auto builder = constructors.at(key);
        return builder(params);
      }
  };

  ScalarBasisFactory& ScalarBasisFactory::instance() {
    static ScalarBasisFactory inst; //This line is supposed to run only once
    return inst;
  }

  /**
   * Use this method to add new basis function types to this factory class.
   * @brief Registers a new basis function type.
   * @since version 0.1.1
   */
  void ScalarBasisFactory::reg(const std::string& type, const ConstructorType& func) {
    _impl->constructors[type] = func;
  }

  std::shared_ptr<ScalarBasisFun> ScalarBasisFactory::construct(const json& obj) const {
    return _impl->construct(obj);
  }

  ScalarBasisFactory::ScalarBasisFactory() {
    _impl = unique_ptr<Impl>( new Impl() );
  }

  using json = nlohmann::json;

  /**
   * Loads a FullProMP object from a text file in JSON format. See json2full_promp for information
   * about the required format.
   * @brief Loads a FullProMP object from the file with the given name
   */
  FullProMP load_full_promp(const std::string& file_name) {
    ifstream in(file_name);
    json obj;
    in >> obj;
    return json2full_promp(obj);
  }
};
