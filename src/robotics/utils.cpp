/**
 * @file
 * In this file you can find basic utilities for linear algebra, basic object loading
 * from JSON and other robotics utilities.
 */

#include "robotics/utils.hpp"
#include <exception>
#include <chrono>

using namespace arma;

namespace robotics {
  /**
   * @brief Converts a JSON object to an arma::mat object
   * @since version 0.0.1
   */
  arma::mat json2mat(const nlohmann::json& obj) {
    int n = obj.size(), m = obj[0].size();
    arma::mat ans(n,m,arma::fill::zeros);
    for (int i=0; i<n; i++) {
      for (int j=0; j<m; j++) {
        ans(i,j) = obj[i][j];
      }
    }
    return ans;
  }

  /**
   * @brief Converts a JSON object to an arma::vec object
   * @since version 0.0.1
   */
  arma::vec json2vec(const nlohmann::json& obj) {
    int n = obj.size();
    arma::vec ans(n);
    for (int i=0; i<n; i++) {
      ans[i] = obj[i];
    }
    return ans;
  }

  /**
   * @brief Converts an arma::vec to a JSON object
   */
  nlohmann::json vec2json(const vec& v) {
    std::vector<double> stdvec = conv_to<std::vector<double>>::from(v);
    nlohmann::json ret(stdvec);
    return ret;
  }

  /**
   * @brief Converts an arma::mat to a JSON object
   */
  nlohmann::json mat2json(const mat& m) {
    std::vector<std::vector<double>> stdvecs;
    for(int i = 0; i < m.n_rows; i++) {
      std::vector<double> stdrow = conv_to<std::vector<double>>::from(
            m.row(i));
      stdvecs.push_back(stdrow);
    }
    nlohmann::json ret(stdvecs);
    return ret;
  }

  /**
   * @brief Creates a block diagonal matrix from the given matrices
   * @since version 0.0.1
   */
  arma::mat block_diag(const std::vector<arma::mat>& mat_list) {
    unsigned int n_rows = 0, n_cols = 0;
    for (const auto& m : mat_list) {
      n_rows += m.n_rows;
      n_cols += m.n_cols;
    }
    mat ans(n_rows, n_cols, fill::zeros);
    unsigned int o_rows = 0, o_cols = 0;
    for (const auto& m : mat_list) {
      for (unsigned int i=0; i<m.n_rows; i++) {
        for (unsigned int j=0; j<m.n_cols; j++) {
          ans(i+o_rows,j+o_cols) = m(i,j);
        }
      }
      o_rows += m.n_rows;
      o_cols += m.n_cols;
    }
    return ans;
  }

  /**
   * @brief Creates a matrix concatenating vertically the given matrices
   * @since version 0.0.1
   */
  arma::mat mat_concat_vertical(const std::vector<arma::mat>& mat_list) {
    if (mat_list.size() == 0) 
      throw std::domain_error("Vector of matrices to concatenate can not be empty");
    unsigned int n_rows = 0, n_cols = mat_list[0].n_cols;
    for (const auto& m : mat_list) {
      if (m.n_cols != n_cols)
        throw std::domain_error("The number of columns of the matrices must match to concatenate vertically");
      n_rows += m.n_rows;
    }
    mat ans(n_rows, n_cols, fill::zeros);
    unsigned int o_rows = 0;
    for (const auto& m : mat_list) {
      for (unsigned int i=0; i<m.n_rows; i++) {
        for (unsigned int j=0; j<m.n_cols; j++) {
          ans(i+o_rows,j) = m(i,j);
        }
      }
      o_rows += m.n_rows;
    }
    return ans;
  }

  /**
   * @brief Projects a 3D point into the image plane with the given projection matrix
   */
  pt2d stereo_project(const pt3d& obs3d, const arma::mat& proj_mat) {
    vec h_p3d {obs3d[0], obs3d[1], obs3d[2], 1.0};
    vec h_p2d = proj_mat*h_p3d;
    pt2d ans{ h_p2d[0]/h_p2d[2], h_p2d[1]/h_p2d[2] };
    return ans;
  }

  /**
   * @brief Returns the re-projection error in pixels with the given projection matrix
   */
  double stereo_proj_error(const pt3d& obs3d, const pt2d& obs2d, const arma::mat& proj_mat) {
    pt2d pr2d = stereo_project(obs3d, proj_mat);
    vec diff { obs2d[0]-pr2d[0], obs2d[1]-pr2d[1] };
    return arma::norm(diff);
  }

  /**
   * Computes the camera calibration matrix from a set of 2d-3d correspondances. If the given
   * point lists are not of the same size a exception is thrown.
   * @brief Return the camera calibration matrix given a set of 2d, 3d point correspondances
   */
  arma::mat calibrate_camera(const std::vector<pt2d>& obs2d, const std::vector<pt3d>& obs3d) {
    if (obs2d.size() != obs3d.size()) 
      throw std::length_error("The number of 2d-3d correspondances must match for camera calibration");
    unsigned int N = obs3d.size();
    mat A(2*N, 11, fill::zeros);
    vec b(2*N);
    for (unsigned int i=0; i<N; i++) {
      for (int j=0; j<2; j++) {
        b[2*i + j] = obs2d[i][j];
        for (int k=0; k<3; k++) {
          A(2*i + j, 4*j + k) = obs3d[i][k];
          A(2*i + j, 8 + k) = -(obs3d[i][k]*obs2d[i][j]);
        }
        A(2*i + j, 4*j + 3) = 1;
      }
    }
    vec values = solve(A,b);
    mat ans(3,4);
    for (int i=0; i<11; i++) ans(i/4,i%4) = values[i];
    ans(2,3) = 1.0;
    return ans;
  }

  /**
   * @brief Return a 3D position of a point given several 2D camera observations and calibration matrices
   * @param[in] calib_mat Dictionary mapping camera index to its calibration matrix
   * @param[in] obs2d Vector of 2D camera observations. Each element of the vector is a pair containing
   * the camera index and the 2D observation on that camera.
   */
  pt3d stereo_vision(const std::unordered_map<unsigned int, arma::mat>& calib_mat, 
      const std::vector<std::pair<unsigned int, pt2d>>& obs2d) {
    if (obs2d.size() < 2) throw std::domain_error("Two or more 2d points are required for stereo vision");
    unsigned int N = obs2d.size();
    mat A(2*N,3);
    vec b(2*N);
    for (unsigned int i=0; i<N; i++) {
      auto c_it = calib_mat.find(obs2d[i].first);
      if (c_it == calib_mat.end())
        throw std::invalid_argument("No calibration matrix provided for a given camera observation");
      const mat& C = c_it->second;
      const pt2d& obs = obs2d[i].second;
      for (int j=0; j<2; j++) {
        for (int k=0; k<3; k++) {
          A(2*i+j,k) = C(j,k) - C(2,k)*obs[j];
        }
        b[2*i+j] = C(2,3)*obs[j] - C(j,3);
      }
    }
    vec values = solve(A,b);
    pt3d ans;
    std::copy(values.begin(), values.end(), ans.begin());
    return ans;
  }

  pt3d stereo_vision(const std::vector<arma::mat>& proj_mat, const std::vector<pt2d>& pts2d) {
    if (proj_mat.size() != pts2d.size()) 
      throw std::domain_error("The number of 2D points and calibration matrices do not match");
    std::unordered_map<unsigned int, arma::mat> _calib_mat;
    std::vector<std::pair<unsigned int, pt2d>> _obs2d;
    for (unsigned int i=0; i<proj_mat.size(); i++) {
      _calib_mat[i] = proj_mat[i];
      _obs2d.push_back( make_pair(i, pts2d[i]) );
    }
    return stereo_vision(_calib_mat, _obs2d);
  }

  std::vector<std::pair<unsigned int, pt2d>> stereo_max_inlier_set(
      const std::unordered_map<unsigned int, arma::mat>& calib_mat, 
      const std::vector<std::pair<unsigned int, pt2d>>& obs2d,
      double max_pix_error) {
    if (obs2d.size() < 2) throw std::domain_error("Two or more 2d points are required for stereo vision");
    std::vector<std::pair<unsigned int, pt2d>> ans;
    double max_err=0.0;
    for (unsigned int i=0; i<(obs2d.size()-1); i++) {
      for (unsigned int j=i+1; j<obs2d.size(); j++) {
        std::vector<std::pair<unsigned int, pt2d>> ptset{obs2d[i],obs2d[j]};
        pt3d candidate = stereo_vision(calib_mat, ptset);
        ptset.clear();
        double s_err=0.0;
        for (unsigned int k=0; k<obs2d.size(); k++) {
          double p_err = stereo_proj_error(candidate, obs2d[k].second, calib_mat.at(obs2d[k].first)); 
          if (p_err < max_pix_error) {
            ptset.push_back(obs2d[k]);
            s_err += p_err;
          }
        }
        if ( ptset.size() > ans.size() || ((ptset.size()==ans.size()) && (s_err<max_err)) ) {
          swap(ans, ptset);
          max_err = s_err;
        }
      }
    }
    return ans;
  }
};
