module ac_constants_mod
use kind_mod, only: dp
implicit none
private
public :: ac_pi, ac_e

real(dp), parameter :: ac_pi = 3.1415926535897932384626433832795_dp
real(dp), parameter :: ac_e = 2.7182818284590452353602874713527_dp

end module ac_constants_mod
