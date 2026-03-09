module ac_random_support
use kind_mod, only: dp
implicit none
private
public :: ac_random_state, ac_random_init, ac_gauss

real(dp), parameter :: ac_pi = 3.1415926535897932384626433832795d0

logical, save :: ac_have_saved = .false.
real(dp), save :: ac_saved_value = 0.0d0

type :: ac_random_state
    integer :: seed = 0
end type ac_random_state

contains

function ac_random_init(seed) result(state)
! Initialize the shared random-number generator state.
! seed: integer seed used to populate random_seed.
integer, intent(in) :: seed
type(ac_random_state) :: state
integer :: n, i
integer, allocatable :: seed_values(:)

call random_seed(size=n)
allocate(seed_values(n))
do i = 1, n
    seed_values(i) = modulo(seed + 104729 * i, huge(1))
    if (seed_values(i) <= 0) seed_values(i) = i
end do
call random_seed(put=seed_values)
deallocate(seed_values)
ac_have_saved = .false.
ac_saved_value = 0.0d0
state%seed = seed
end function ac_random_init

function ac_gauss(state, mean, stddev) result(value)
! Draw one Gaussian variate using the Box-Muller transform.
! state: mutable RNG state placeholder.
! mean: target mean.
! stddev: target standard deviation.
type(ac_random_state), intent(inout) :: state
real(dp), intent(in) :: mean
real(dp), intent(in) :: stddev
real(dp) :: value
real(dp) :: u1, u2, radius, angle, z

if (ac_have_saved) then
    z = ac_saved_value
    ac_have_saved = .false.
else
    call random_number(u1)
    call random_number(u2)
    if (u1 <= 0.0d0) u1 = 1.0d-12
    radius = sqrt(-2.0d0 * log(u1))
    angle = 2.0d0 * ac_pi * u2
    z = radius * cos(angle)
    ac_saved_value = radius * sin(angle)
    ac_have_saved = .true.
end if
value = mean + stddev * z
end function ac_gauss

end module ac_random_support
