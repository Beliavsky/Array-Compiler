module ac_numpy_mod
use kind_mod, only: dp
use ac_random_support, only: ac_random_state, ac_gauss
implicit none
private
public :: ac_array, ac_asarray, ac_ndim, ac_shape_dim, ac_empty2, ac_choice_weighted
public :: ac_choice_no_replace, ac_where, ac_row, ac_set_rows, ac_reshape
public :: ac_multivariate_normal, ac_set_row, ac_take_rows, ac_savetxt
public :: ac_copy, ac_full, ac_eye, ac_column, ac_add_axis, ac_set_column
public :: ac_sum_axis, ac_max_axis, ac_cov_rowvar_false
public :: ac_slogdet_sign, ac_slogdet_logabsdet, ac_inv
public :: ac_einsum_ni_ij_nj_to_n, ac_loadtxt, ac_atleast_2d, ac_argsort
public :: ac_transpose, ac_matmul, ac_float, ac_sub_row, ac_sub_col, ac_mul_col

interface ac_array
    module procedure ac_array_1d_real
    module procedure ac_array_2d_real
end interface

interface ac_asarray
    module procedure ac_asarray_1d_real
    module procedure ac_asarray_2d_real
end interface

interface ac_copy
    module procedure ac_copy_1d_real
    module procedure ac_copy_2d_real
end interface

interface ac_ndim
    module procedure ac_ndim_1d_real
    module procedure ac_ndim_2d_real
    module procedure ac_ndim_1d_int
end interface

interface ac_shape_dim
    module procedure ac_shape_dim_1d_real
    module procedure ac_shape_dim_2d_real
    module procedure ac_shape_dim_1d_int
end interface

interface ac_row
    module procedure ac_row_2d_real
end interface

interface ac_take_rows
    module procedure ac_take_rows_2d_real
end interface

interface ac_column
    module procedure ac_column_2d_real
end interface

interface ac_reshape
    module procedure ac_reshape_1d_to_2d_real
    module procedure ac_reshape_2d_to_1d_real
end interface

interface ac_atleast_2d
    module procedure ac_atleast_2d_real_1d
    module procedure ac_atleast_2d_real_2d
end interface

interface ac_transpose
    module procedure ac_transpose_2d_real
end interface

interface ac_matmul
    module procedure ac_matmul_2d_2d_real
end interface

interface ac_float
    module procedure ac_float_scalar
    module procedure ac_float_1x1
end interface

contains

pure function ac_array_1d_real(x) result(y)
real(dp), intent(in) :: x(:)
real(dp), allocatable :: y(:)
y = x
end function ac_array_1d_real

pure function ac_array_2d_real(x) result(y)
real(dp), intent(in) :: x(:, :)
real(dp), allocatable :: y(:, :)
y = x
end function ac_array_2d_real

pure function ac_asarray_1d_real(x) result(y)
real(dp), intent(in) :: x(:)
real(dp), allocatable :: y(:)
y = x
end function ac_asarray_1d_real

pure function ac_asarray_2d_real(x) result(y)
real(dp), intent(in) :: x(:, :)
real(dp), allocatable :: y(:, :)
y = x
end function ac_asarray_2d_real

pure function ac_copy_1d_real(x) result(y)
real(dp), intent(in) :: x(:)
real(dp), allocatable :: y(:)
y = x
end function ac_copy_1d_real

pure function ac_copy_2d_real(x) result(y)
real(dp), intent(in) :: x(:, :)
real(dp), allocatable :: y(:, :)
y = x
end function ac_copy_2d_real

pure integer function ac_ndim_1d_real(x)
real(dp), intent(in) :: x(:)
ac_ndim_1d_real = 1
end function ac_ndim_1d_real

pure integer function ac_ndim_2d_real(x)
real(dp), intent(in) :: x(:, :)
ac_ndim_2d_real = 2
end function ac_ndim_2d_real

pure integer function ac_ndim_1d_int(x)
integer, intent(in) :: x(:)
ac_ndim_1d_int = 1
end function ac_ndim_1d_int

pure integer function ac_shape_dim_1d_real(x, dim_index)
real(dp), intent(in) :: x(:)
integer, intent(in) :: dim_index
if (dim_index == 1) then
    ac_shape_dim_1d_real = size(x, 1)
else
    ac_shape_dim_1d_real = 1
end if
end function ac_shape_dim_1d_real

pure integer function ac_shape_dim_2d_real(x, dim_index)
real(dp), intent(in) :: x(:, :)
integer, intent(in) :: dim_index
ac_shape_dim_2d_real = size(x, dim_index)
end function ac_shape_dim_2d_real

pure integer function ac_shape_dim_1d_int(x, dim_index)
integer, intent(in) :: x(:)
integer, intent(in) :: dim_index
if (dim_index == 1) then
    ac_shape_dim_1d_int = size(x, 1)
else
    ac_shape_dim_1d_int = 1
end if
end function ac_shape_dim_1d_int

pure function ac_empty2(n, m) result(x)
integer, intent(in) :: n, m
real(dp), allocatable :: x(:, :)
allocate(x(n, m))
end function ac_empty2

pure function ac_full(n, value) result(x)
integer, intent(in) :: n
real(dp), intent(in) :: value
real(dp), allocatable :: x(:)
allocate(x(n))
x = value
end function ac_full

pure function ac_eye(n) result(x)
integer, intent(in) :: n
real(dp), allocatable :: x(:, :)
integer :: i
allocate(x(n, n))
x = 0.0_dp
do i = 1, n
    x(i, i) = 1.0_dp
end do
end function ac_eye

function ac_choice_weighted(state, npop, nsamp, p) result(z)
type(ac_random_state), intent(inout) :: state
integer, intent(in) :: npop, nsamp
real(dp), intent(in) :: p(:)
integer, allocatable :: z(:)
real(dp) :: u, cumulative
integer :: i, j

allocate(z(nsamp))
do i = 1, nsamp
    call random_number(u)
    cumulative = 0.0_dp
    z(i) = npop - 1
    do j = 1, min(npop, size(p))
        cumulative = cumulative + p(j)
        if (u <= cumulative) then
            z(i) = j - 1
            exit
        end if
    end do
end do
end function ac_choice_weighted

function ac_choice_no_replace(state, npop, nsamp) result(z)
type(ac_random_state), intent(inout) :: state
integer, intent(in) :: npop, nsamp
integer, allocatable :: z(:)
integer, allocatable :: pool(:)
real(dp) :: u
integer :: i, pick, remaining, temp

allocate(z(nsamp), pool(npop))
pool = [(i - 1, i = 1, npop)]
remaining = npop
do i = 1, nsamp
    call random_number(u)
    pick = 1 + int(u * remaining)
    if (pick > remaining) pick = remaining
    z(i) = pool(pick)
    temp = pool(pick)
    pool(pick) = pool(remaining)
    pool(remaining) = temp
    remaining = remaining - 1
end do
end function ac_choice_no_replace

function ac_where(mask) result(idx)
logical, intent(in) :: mask(:)
integer, allocatable :: idx(:)
integer :: count_true, i, pos

count_true = count(mask)
allocate(idx(count_true))
pos = 0
do i = 1, size(mask)
    if (mask(i)) then
        pos = pos + 1
        idx(pos) = i - 1
    end if
end do
end function ac_where

pure function ac_row_2d_real(x, row_index) result(y)
real(dp), intent(in) :: x(:, :)
integer, intent(in) :: row_index
real(dp), allocatable :: y(:)
y = x(row_index + 1, :)
end function ac_row_2d_real

pure function ac_column_2d_real(x, col_index) result(y)
real(dp), intent(in) :: x(:, :)
integer, intent(in) :: col_index
real(dp), allocatable :: y(:)
y = x(:, col_index + 1)
end function ac_column_2d_real

subroutine ac_set_rows(x, idx, values)
real(dp), intent(inout) :: x(:, :)
integer, intent(in) :: idx(:)
real(dp), intent(in) :: values(:, :)
integer :: i

do i = 1, size(idx)
    x(idx(i) + 1, :) = values(i, :)
end do
end subroutine ac_set_rows

subroutine ac_set_row(x, row_index, values)
real(dp), intent(inout) :: x(:, :)
integer, intent(in) :: row_index
real(dp), intent(in) :: values(:)

x(row_index + 1, :) = values
end subroutine ac_set_row

subroutine ac_set_column(x, col_index, values)
real(dp), intent(inout) :: x(:, :)
integer, intent(in) :: col_index
real(dp), intent(in) :: values(:)

x(:, col_index + 1) = values
end subroutine ac_set_column

pure function ac_take_rows_2d_real(x, idx) result(y)
real(dp), intent(in) :: x(:, :)
integer, intent(in) :: idx(:)
real(dp), allocatable :: y(:, :)
integer :: i

allocate(y(size(idx), size(x, 2)))
do i = 1, size(idx)
    y(i, :) = x(idx(i) + 1, :)
end do
end function ac_take_rows_2d_real

pure function ac_reshape_1d_to_2d_real(x, n, m) result(y)
real(dp), intent(in) :: x(:)
integer, intent(in) :: n, m
real(dp), allocatable :: y(:, :)
integer :: i, j

allocate(y(n, m))
do j = 1, m
    do i = 1, n
        y(i, j) = x((i - 1) * m + j)
    end do
end do
end function ac_reshape_1d_to_2d_real

pure function ac_reshape_2d_to_1d_real(x, n) result(y)
real(dp), intent(in) :: x(:, :)
integer, intent(in) :: n
real(dp), allocatable :: y(:)
integer :: i, j, k

allocate(y(n))
k = 0
do i = 1, size(x, 1)
    do j = 1, size(x, 2)
        k = k + 1
        if (k <= n) y(k) = x(i, j)
    end do
end do
end function ac_reshape_2d_to_1d_real

pure function ac_add_axis(x) result(y)
real(dp), intent(in) :: x(:)
real(dp), allocatable :: y(:, :)
integer :: i

allocate(y(size(x), 1))
do i = 1, size(x)
    y(i, 1) = x(i)
end do
end function ac_add_axis

pure function ac_sum_axis(x, axis) result(y)
real(dp), intent(in) :: x(:, :)
integer, intent(in) :: axis
real(dp), allocatable :: y(:)

if (axis == 0) then
    allocate(y(size(x, 2)))
    y = sum(x, dim=1)
else
    allocate(y(size(x, 1)))
    y = sum(x, dim=2)
end if
end function ac_sum_axis

pure function ac_max_axis(x, axis) result(y)
real(dp), intent(in) :: x(:, :)
integer, intent(in) :: axis
real(dp), allocatable :: y(:)

if (axis == 0) then
    allocate(y(size(x, 2)))
    y = maxval(x, dim=1)
else
    allocate(y(size(x, 1)))
    y = maxval(x, dim=2)
end if
end function ac_max_axis

pure function ac_transpose_2d_real(x) result(y)
real(dp), intent(in) :: x(:, :)
real(dp), allocatable :: y(:, :)
y = transpose(x)
end function ac_transpose_2d_real

pure function ac_matmul_2d_2d_real(a, b) result(c)
real(dp), intent(in) :: a(:, :)
real(dp), intent(in) :: b(:, :)
real(dp), allocatable :: c(:, :)
c = matmul(a, b)
end function ac_matmul_2d_2d_real

pure function ac_sub_row(x, row) result(y)
real(dp), intent(in) :: x(:, :)
real(dp), intent(in) :: row(:)
real(dp), allocatable :: y(:, :)
integer :: j

allocate(y(size(x, 1), size(x, 2)))
y = x
do j = 1, size(x, 2)
    y(:, j) = y(:, j) - row(j)
end do
end function ac_sub_row

pure function ac_sub_col(x, col) result(y)
real(dp), intent(in) :: x(:, :)
real(dp), intent(in) :: col(:)
real(dp), allocatable :: y(:, :)
integer :: i

allocate(y(size(x, 1), size(x, 2)))
y = x
do i = 1, size(x, 1)
    y(i, :) = y(i, :) - col(i)
end do
end function ac_sub_col

pure function ac_mul_col(x, col) result(y)
real(dp), intent(in) :: x(:, :)
real(dp), intent(in) :: col(:)
real(dp), allocatable :: y(:, :)
integer :: i

allocate(y(size(x, 1), size(x, 2)))
y = x
do i = 1, size(x, 1)
    y(i, :) = y(i, :) * col(i)
end do
end function ac_mul_col

function ac_multivariate_normal(mean, cov, nsamp) result(x)
real(dp), intent(in) :: mean(:)
real(dp), intent(in) :: cov(:, :)
integer, intent(in) :: nsamp
real(dp), allocatable :: x(:, :)
real(dp), allocatable :: chol(:, :), z(:)
integer :: i

chol = ac_cholesky(cov)
allocate(x(nsamp, size(mean)))
allocate(z(size(mean)))
do i = 1, nsamp
    call ac_standard_normal_vec(z)
    x(i, :) = mean + matmul(chol, z)
end do
end function ac_multivariate_normal

function ac_cov_rowvar_false(x) result(cov)
real(dp), intent(in) :: x(:, :)
real(dp), allocatable :: cov(:, :)
real(dp), allocatable :: xc(:, :)
real(dp), allocatable :: mu(:)
integer :: n, d, j

n = size(x, 1)
d = size(x, 2)
allocate(mu(d), xc(n, d), cov(d, d))
mu = sum(x, dim=1) / real(n, dp)
xc = x
do j = 1, d
    xc(:, j) = xc(:, j) - mu(j)
end do
if (n > 1) then
    cov = matmul(transpose(xc), xc) / real(n - 1, dp)
else
    cov = 0.0_dp
end if
end function ac_cov_rowvar_false

function ac_slogdet_sign(a) result(sign_value)
real(dp), intent(in) :: a(:, :)
real(dp) :: sign_value
real(dp), allocatable :: lu(:, :)
integer :: i, swap_count

call ac_lu_factor(a, lu, swap_count)
sign_value = merge(1.0_dp, -1.0_dp, modulo(swap_count, 2) == 0)
do i = 1, size(lu, 1)
    sign_value = sign_value * sign(1.0_dp, lu(i, i))
end do
end function ac_slogdet_sign

function ac_slogdet_logabsdet(a) result(logabsdet)
real(dp), intent(in) :: a(:, :)
real(dp) :: logabsdet
real(dp), allocatable :: lu(:, :)
integer :: i

call ac_lu_factor(a, lu)
logabsdet = 0.0_dp
do i = 1, size(lu, 1)
    logabsdet = logabsdet + log(abs(lu(i, i)))
end do
end function ac_slogdet_logabsdet

function ac_inv(a) result(inv)
real(dp), intent(in) :: a(:, :)
real(dp), allocatable :: inv(:, :)
real(dp), allocatable :: aug(:, :)
real(dp) :: pivot_abs, factor
integer :: n, i, j, pivot_row

n = size(a, 1)
allocate(inv(n, n), aug(n, 2 * n))
aug = 0.0_dp
aug(:, 1:n) = a
do i = 1, n
    aug(i, n + i) = 1.0_dp
end do

do i = 1, n
    pivot_row = i
    pivot_abs = abs(aug(i, i))
    do j = i + 1, n
        if (abs(aug(j, i)) > pivot_abs) then
            pivot_abs = abs(aug(j, i))
            pivot_row = j
        end if
    end do
    if (pivot_abs < 1.0d-14) error stop "singular matrix in ac_inv"
    if (pivot_row /= i) call ac_swap_rows(aug, i, pivot_row)
    aug(i, :) = aug(i, :) / aug(i, i)
    do j = 1, n
        if (j /= i) then
            factor = aug(j, i)
            aug(j, :) = aug(j, :) - factor * aug(i, :)
        end if
    end do
end do
inv = aug(:, n + 1:2 * n)
end function ac_inv

pure function ac_einsum_ni_ij_nj_to_n(diff, inv_cov, diff2) result(y)
real(dp), intent(in) :: diff(:, :)
real(dp), intent(in) :: inv_cov(:, :)
real(dp), intent(in) :: diff2(:, :)
real(dp), allocatable :: y(:)
integer :: i

allocate(y(size(diff, 1)))
do i = 1, size(diff, 1)
    y(i) = dot_product(diff(i, :), matmul(inv_cov, diff2(i, :)))
end do
end function ac_einsum_ni_ij_nj_to_n

function ac_loadtxt(path) result(x)
character(len=*), intent(in) :: path
real(dp), allocatable :: x(:, :)
character(len=4096) :: line
integer :: unit, ios, nrow, ncol, i

nrow = 0
ncol = 0
open(newunit=unit, file=trim(path), status="old", action="read")
do
    read(unit, '(A)', iostat=ios) line
    if (ios /= 0) exit
    if (len_trim(line) == 0) cycle
    nrow = nrow + 1
    if (ncol == 0) ncol = ac_count_fields(line)
end do
rewind(unit)
allocate(x(nrow, ncol))
i = 0
do
    read(unit, '(A)', iostat=ios) line
    if (ios /= 0) exit
    if (len_trim(line) == 0) cycle
    i = i + 1
    read(line, *) x(i, :)
end do
close(unit)
end function ac_loadtxt

pure function ac_atleast_2d_real_1d(x) result(y)
real(dp), intent(in) :: x(:)
real(dp), allocatable :: y(:, :)
allocate(y(1, size(x)))
y(1, :) = x
end function ac_atleast_2d_real_1d

pure function ac_atleast_2d_real_2d(x) result(y)
real(dp), intent(in) :: x(:, :)
real(dp), allocatable :: y(:, :)
y = x
end function ac_atleast_2d_real_2d

function ac_argsort(x) result(idx)
real(dp), intent(in) :: x(:)
integer, allocatable :: idx(:)
integer :: i, j, tmp

allocate(idx(size(x)))
idx = [(i - 1, i = 1, size(x))]
do i = 1, size(idx) - 1
    do j = i + 1, size(idx)
        if (x(idx(j) + 1) < x(idx(i) + 1)) then
            tmp = idx(i)
            idx(i) = idx(j)
            idx(j) = tmp
        end if
    end do
end do
end function ac_argsort

pure function ac_float_scalar(x) result(y)
real(dp), intent(in) :: x
real(dp) :: y
y = x
end function ac_float_scalar

pure function ac_float_1x1(x) result(y)
real(dp), intent(in) :: x(:, :)
real(dp) :: y
y = x(1, 1)
end function ac_float_1x1

subroutine ac_savetxt(path, x, fmt)
character(len=*), intent(in) :: path
real(dp), intent(in) :: x(:, :)
character(len=*), intent(in) :: fmt
integer :: unit, i, j
character(len=64) :: item_fmt
integer :: decimals, ios

item_fmt = trim(adjustl(fmt))
if (len_trim(item_fmt) >= 4 .and. item_fmt(1:1) == "%" .and. item_fmt(len_trim(item_fmt):len_trim(item_fmt)) == "f") then
    read(item_fmt(3:len_trim(item_fmt)-1), *, iostat=ios) decimals
    if (ios == 0) then
        write(item_fmt, '("(f0.", i0, ")")') decimals
    end if
end if
open(newunit=unit, file=trim(path), status="replace", action="write")
do i = 1, size(x, 1)
    do j = 1, size(x, 2)
        if (j > 1) write(unit, "(a)", advance="no") " "
        write(unit, "(" // trim(item_fmt) // ")", advance="no") x(i, j)
    end do
    write(unit, *)
end do
close(unit)
end subroutine ac_savetxt

subroutine ac_standard_normal_vec(z)
real(dp), intent(out) :: z(:)
type(ac_random_state) :: dummy_state
integer :: i

dummy_state%seed = 0
do i = 1, size(z)
    z(i) = ac_gauss(dummy_state, 0.0_dp, 1.0_dp)
end do
end subroutine ac_standard_normal_vec

pure integer function ac_count_fields(line)
character(len=*), intent(in) :: line
integer :: i
logical :: in_field

ac_count_fields = 0
in_field = .false.
do i = 1, len_trim(line)
    if (line(i:i) /= ' ' .and. line(i:i) /= char(9)) then
        if (.not. in_field) then
            ac_count_fields = ac_count_fields + 1
            in_field = .true.
        end if
    else
        in_field = .false.
    end if
end do
end function ac_count_fields

function ac_cholesky(a) result(l)
real(dp), intent(in) :: a(:, :)
real(dp), allocatable :: l(:, :)
integer :: n, i, j, k
real(dp) :: s

n = size(a, 1)
allocate(l(n, n))
l = 0.0_dp
do i = 1, n
    do j = 1, i
        s = a(i, j)
        do k = 1, j - 1
            s = s - l(i, k) * l(j, k)
        end do
        if (i == j) then
            if (s <= 0.0_dp) error stop "covariance must be positive definite"
            l(i, j) = sqrt(s)
        else
            l(i, j) = s / l(j, j)
        end if
    end do
end do
end function ac_cholesky

subroutine ac_lu_factor(a, lu, swap_count, piv)
real(dp), intent(in) :: a(:, :)
real(dp), allocatable, intent(out) :: lu(:, :)
integer, intent(out), optional :: swap_count
integer, allocatable, intent(out), optional :: piv(:)
integer :: n, i, j, k, pivot_row, swaps
real(dp) :: pivot_abs
integer, allocatable :: piv_local(:)

n = size(a, 1)
allocate(lu(n, n), piv_local(n))
lu = a
piv_local = [(i, i = 1, n)]
swaps = 0
do k = 1, n - 1
    pivot_row = k
    pivot_abs = abs(lu(k, k))
    do i = k + 1, n
        if (abs(lu(i, k)) > pivot_abs) then
            pivot_abs = abs(lu(i, k))
            pivot_row = i
        end if
    end do
    if (pivot_abs < 1.0d-14) error stop "singular matrix in ac_lu_factor"
    if (pivot_row /= k) then
        call ac_swap_rows(lu, k, pivot_row)
        call ac_swap_ints(piv_local(k), piv_local(pivot_row))
        swaps = swaps + 1
    end if
    do i = k + 1, n
        lu(i, k) = lu(i, k) / lu(k, k)
        do j = k + 1, n
            lu(i, j) = lu(i, j) - lu(i, k) * lu(k, j)
        end do
    end do
end do
if (abs(lu(n, n)) < 1.0d-14) error stop "singular matrix in ac_lu_factor"
if (present(swap_count)) swap_count = swaps
if (present(piv)) then
    allocate(piv(n))
    piv = piv_local
end if
end subroutine ac_lu_factor

subroutine ac_lu_solve_inplace(lu, b)
real(dp), intent(in) :: lu(:, :)
real(dp), intent(inout) :: b(:)
integer :: n, i, j

n = size(lu, 1)
do i = 2, n
    do j = 1, i - 1
        b(i) = b(i) - lu(i, j) * b(j)
    end do
end do
do i = n, 1, -1
    do j = i + 1, n
        b(i) = b(i) - lu(i, j) * b(j)
    end do
    b(i) = b(i) / lu(i, i)
end do
end subroutine ac_lu_solve_inplace

subroutine ac_swap_rows(a, i, j)
real(dp), intent(inout) :: a(:, :)
integer, intent(in) :: i, j
real(dp) :: tmp(size(a, 2))

tmp = a(i, :)
a(i, :) = a(j, :)
a(j, :) = tmp
end subroutine ac_swap_rows

subroutine ac_swap_scalars(x, y)
real(dp), intent(inout) :: x, y
real(dp) :: tmp

tmp = x
x = y
y = tmp
end subroutine ac_swap_scalars

subroutine ac_swap_ints(x, y)
integer, intent(inout) :: x, y
integer :: tmp

tmp = x
x = y
y = tmp
end subroutine ac_swap_ints

end module ac_numpy_mod
