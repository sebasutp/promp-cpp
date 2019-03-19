/**
 * @file
 * In this file you can find classes and methods useful to convert JSON objects to objects
 * of the robotics namespace. It is very important to state that these functions and classes
 * expect a JSON object with a certain structure for every type. If you pass as parameter
 * a JSON object without the right structure it will result in undefined behaviour.
 */

#ifndef ROBOTICS_JSON_FACTORIES_H
#define ROBOTICS_JSON_FACTORIES_H

#include "robotics/basis_functions.hpp"
#include "robotics/full_promp.hpp"
#include <memory>
#include <functional>
#include <string>
#include <json.hpp>

namespace robotics {

  std::shared_ptr<ScalarBasisFun> json2ScalarPolyBasis(const nlohmann::json& params);
  std::shared_ptr<ScalarBasisFun> json2ScalarGaussBasis(const nlohmann::json& params);
  std::shared_ptr<ScalarBasisFun> json2ScalarCombBasis(const nlohmann::json& basis);

  ProMP json2basic_promp(const nlohmann::json& params);
  FullProMP json2full_promp(const nlohmann::json& params);

  FullProMP load_full_promp(const std::string& file_name);

  /**
   * Factory class for Scalar Basis Functions from JSON objects. The passed JSON object to
   * the construct method should have two properties, called "type" and "params". The property
   * "type" must contain a string that corresponds to a key name for the type of the basis
   * function type that the user wants to instantiate. And the property "params" is a JSON
   * object containing a set of params that will be simply passed on to the particular
   * JSON constructor for each type. If you want to register new constructors you should call
   * the method reg with a new key type and particular constructor function.
   * @brief Scalar basis functions factory class
   * @since version 0.1.1
   */
  class ScalarBasisFactory {
    public:
      using json = nlohmann::json;
      using ConstructorType = std::function<std::shared_ptr<ScalarBasisFun>(const json& obj)>;
      static ScalarBasisFactory& instance();
      void reg(const std::string& type, const ConstructorType& func);
      std::shared_ptr<ScalarBasisFun> construct(const json& obj) const;
    private:
      ScalarBasisFactory();
      class Impl;
      std::unique_ptr<Impl> _impl;
  };

};

#endif
