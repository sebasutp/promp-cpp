
#include <robotics/utils/chrono.hpp>
#include <sys/time.h>

using namespace std;

namespace robotics {

  class Chronometer::Impl {
    public:
      timeval start_time;

      void restart() {
        gettimeofday(&start_time, NULL);
      }

      Impl() {
        start_time.tv_sec = 0;
        start_time.tv_usec = 0;
        //restart();
      }

      double time() const {
        /*Linux only code*/
        timeval stamp;
        gettimeofday(&stamp, NULL);
        return (stamp.tv_sec - start_time.tv_sec) + 1e-6*(stamp.tv_usec - start_time.tv_usec);
        /* end of linux only code */
      }
  };

  Chronometer::Chronometer() {
    _impl = unique_ptr<Impl>( new Impl );
  }

  Chronometer::~Chronometer() = default;

  /**
   * Resets the chronometer time to zero.
   */
  void Chronometer::restart() {
    _impl->restart();
  }
      
  /**
   * Returns the current chronometer time
   */
  double Chronometer::operator()() const {
    return _impl->time();
  }

};
