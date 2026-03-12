module ac_lapack_mod
use kind_mod, only: dp
implicit none
private
public :: ac_eig, ac_svd

contains

subroutine ac_eig(a, w, v)
! Compute right eigenpairs of a real square matrix.
! Only real-spectrum matrices are supported in this path.
real(dp), intent(in) :: a(:, :)
real(dp), allocatable, intent(out) :: w(:), v(:, :)
real(dp), allocatable :: ac(:, :), wr(:), wi(:), vr(:, :), vl_dummy(:, :), work(:)
integer :: n, info, lwork

interface
    subroutine dgeev(jobvl, jobvr, n, a, lda, wr, wi, vl, ldvl, vr, ldvr, work, lwork, info)
        character(len=1), intent(in) :: jobvl, jobvr
        integer, intent(in) :: n, lda, ldvl, ldvr, lwork
        integer, intent(out) :: info
        double precision, intent(inout) :: a(lda, *)
        double precision, intent(out) :: wr(*), wi(*), vl(ldvl, *), vr(ldvr, *), work(*)
    end subroutine dgeev
end interface

n = size(a, 1)
if (size(a, 2) /= n) error stop "ac_eig: matrix must be square"
allocate(ac(n, n), source=a)
allocate(wr(n), wi(n), vr(n, n), vl_dummy(1, 1), work(1))
lwork = -1
call dgeev("N", "V", n, ac, n, wr, wi, vl_dummy, 1, vr, n, work, lwork, info)
if (info /= 0) error stop "ac_eig: dgeev workspace query failed"
lwork = max(1, int(work(1)))
deallocate(work)
allocate(work(lwork))
call dgeev("N", "V", n, ac, n, wr, wi, vl_dummy, 1, vr, n, work, lwork, info)
if (info /= 0) error stop "ac_eig: dgeev failed"
if (maxval(abs(wi)) > 1.0e-12_dp) error stop "ac_eig: complex eigenvalues not supported"
w = wr
v = vr
end subroutine ac_eig

subroutine ac_svd(a, u, s, vt)
! Compute the full SVD of a real matrix.
real(dp), intent(in) :: a(:, :)
real(dp), allocatable, intent(out) :: u(:, :), s(:), vt(:, :)
real(dp), allocatable :: ac(:, :), work(:)
integer :: m, n, k, info, lwork

interface
    subroutine dgesvd(jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork, info)
        character(len=1), intent(in) :: jobu, jobvt
        integer, intent(in) :: m, n, lda, ldu, ldvt, lwork
        integer, intent(out) :: info
        double precision, intent(inout) :: a(lda, *)
        double precision, intent(out) :: s(*), u(ldu, *), vt(ldvt, *), work(*)
    end subroutine dgesvd
end interface

m = size(a, 1)
n = size(a, 2)
k = min(m, n)
allocate(ac(m, n), source=a)
allocate(u(m, m), s(k), vt(n, n), work(1))
lwork = -1
call dgesvd("A", "A", m, n, ac, m, s, u, m, vt, n, work, lwork, info)
if (info /= 0) error stop "ac_svd: dgesvd workspace query failed"
lwork = max(1, int(work(1)))
deallocate(work)
allocate(work(lwork))
call dgesvd("A", "A", m, n, ac, m, s, u, m, vt, n, work, lwork, info)
if (info /= 0) error stop "ac_svd: dgesvd failed"
end subroutine ac_svd

end module ac_lapack_mod
