#ifndef ROBOTICS_UTILS_H
#define ROBOTICS_UTILS_H

#include <armadillo>
#include <json.hpp>
#include <vector>
#include <array>
#include <unordered_map>
#include <memory>
#include "utils/promp_utils.hpp"
#include "utils/chrono.hpp"
#include "utils/random.hpp"
#include "utils/math.hpp"

namespace robotics {
  arma::mat json2mat(const nlohmann::json& obj);
  arma::vec json2vec(const nlohmann::json& obj);
  nlohmann::json mat2json(const arma::mat& m);
  nlohmann::json vec2json(const arma::vec& v);
  arma::mat block_diag(const std::vector<arma::mat>& mat_list);
  arma::mat mat_concat_vertical(const std::vector<arma::mat>& mat_list);

  using pt2d = std::array<double,2>;
  using pt3d = std::array<double,3>;
  pt2d stereo_project(const pt3d& obs3d, const arma::mat& proj_mat);
  double stereo_proj_error(const pt3d& obs3d, const pt2d& obs2d, const arma::mat& proj_mat);
  arma::mat calibrate_camera(const std::vector<pt2d>& obs2d, const std::vector<pt3d>& obs3d);
  pt3d stereo_vision(const std::unordered_map<unsigned int, arma::mat>& calib_mat, 
      const std::vector<std::pair<unsigned int, pt2d>>& obs2d);
  pt3d stereo_vision(const std::vector<arma::mat>& proj_mat, const std::vector<pt2d>& pts2d);
  std::vector<std::pair<unsigned int, pt2d>> stereo_max_inlier_set(
      const std::unordered_map<unsigned int, arma::mat>& calib_mat, 
      const std::vector<std::pair<unsigned int, pt2d>>& obs2d,
      double max_pix_error);
};

#endif
