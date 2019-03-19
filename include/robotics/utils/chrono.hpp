#ifndef ROBOTICS_UTILS_CHRONO_HPP
#define ROBOTICS_UTILS_CHRONO_HPP

#include <memory>

namespace robotics {

  /**
   * @brief This class provides a high precision chronometer.
   */
  class Chronometer {
    public:
      Chronometer();
      ~Chronometer();

      void restart();
      double operator()() const;
    private:
      class Impl;
      std::unique_ptr<Impl> _impl;
  };
};

#endif
