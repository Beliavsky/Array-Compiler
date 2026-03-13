module ac_stats_mod
use kind_mod, only: dp
use ac_constants_mod, only: ac_pi
implicit none
private
public :: ac_dnorm, ac_dnorm_log, ac_quantile

contains

pure elemental function ac_dnorm(x, mean, sd) result(y)
real(dp), intent(in) :: x, mean, sd
real(dp) :: y
real(dp) :: logc, z

if (sd <= 0.0_dp) error stop "ac_dnorm requires sd > 0"
logc = -log(sd) - 0.5_dp * log(2.0_dp * ac_pi)
z = (x - mean) / sd
y = exp(logc - 0.5_dp * z * z)
end function ac_dnorm

pure elemental function ac_dnorm_log(x, mean, sd) result(y)
real(dp), intent(in) :: x, mean, sd
real(dp) :: y
real(dp) :: logc, z

if (sd <= 0.0_dp) error stop "ac_dnorm_log requires sd > 0"
logc = -log(sd) - 0.5_dp * log(2.0_dp * ac_pi)
z = (x - mean) / sd
y = logc - 0.5_dp * z * z
end function ac_dnorm_log

pure function ac_quantile(x, probs) result(y)
real(dp), intent(in) :: x(:), probs(:)
real(dp), allocatable :: y(:)
real(dp), allocatable :: work(:)
real(dp) :: h, frac, tmp
integer :: i, j, n, lo, hi

n = size(x)
allocate(y(size(probs)))
if (n == 0) then
    y = 0.0_dp
    return
end if
allocate(work(n))
work = x
do i = 1, n
    do j = i + 1, n
        if (work(j) < work(i)) then
            tmp = work(i)
            work(i) = work(j)
            work(j) = tmp
        end if
    end do
end do
do i = 1, size(probs)
    if (probs(i) <= 0.0_dp) then
        y(i) = work(1)
    else if (probs(i) >= 1.0_dp) then
        y(i) = work(n)
    else
        h = 1.0_dp + (real(n, dp) - 1.0_dp) * probs(i)
        lo = int(floor(h))
        hi = int(ceiling(h))
        frac = h - real(lo, dp)
        y(i) = work(lo) + frac * (work(hi) - work(lo))
    end if
end do
end function ac_quantile

end module ac_stats_mod
