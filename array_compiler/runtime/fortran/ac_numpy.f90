module ac_numpy_mod
use kind_mod, only: dp
use ac_random_support, only: ac_random_state, ac_gauss
use, intrinsic :: ieee_arithmetic, only: ieee_is_finite, ieee_is_nan, ieee_quiet_nan, ieee_value
implicit none
private
public :: ac_array, ac_asarray, ac_ndim, ac_shape_dim, ac_empty, ac_empty2, ac_choice_weighted
public :: ac_choice_no_replace, ac_where, ac_where_select, ac_row, ac_set_rows, &
    & ac_reshape, ac_mask, ac_slice2
public :: ac_multivariate_normal, ac_set_row, ac_take_rows, ac_savetxt, ac_file_exists
public :: ac_copy, ac_full, ac_eye, ac_column, ac_add_axis, ac_set_column
public :: ac_sum_axis, ac_mean_axis, ac_min_axis, ac_max_axis, ac_argmin, ac_argmax, ac_argmin_axis, &
    & ac_argmax_axis, ac_var, ac_std, ac_cov_rowvar_false
public :: ac_slogdet_sign, ac_slogdet_logabsdet, ac_inv, ac_cholesky
public :: ac_einsum_ni_ij_nj_to_n, ac_loadtxt, ac_loadtxt_1d, ac_atleast_2d, ac_argsort
public :: ac_transpose, ac_transpose_perm, ac_swapaxes, ac_matmul, ac_float, ac_sub_row, ac_sub_col, ac_mul_col
public :: ac_reverse, ac_r_concat
public :: ac_roots
public :: ac_normal_vec, ac_uniform_vec, ac_zeros, ac_zeros2, ac_dot, ac_mean, &
    & ac_slice, ac_slice_step, ac_set_slice, ac_fill_slice, ac_clip, ac_any, ac_norm
public :: ac_arange, ac_arange_int, ac_linspace, ac_column_stack, ac_round, &
    & ac_solve_linear, ac_solve_linear_fallback, ac_cumsum, ac_cumprod, ac_diff, ac_gradient
public :: ac_item2, ac_set_item2, ac_pick2, ac_set_pick2, ac_sign, ac_row_axis, ac_astype_int, ac_astype_float
public :: ac_sort, ac_unique, ac_bincount, ac_searchsorted_left, ac_searchsorted_right, ac_repeat, &
    & ac_repeat_axis, ac_tile, ac_diag, ac_diag_k, ac_triu, ac_tril
public :: ac_trace, ac_outer, ac_kron, ac_concatenate_axis0, ac_concatenate_axis1, ac_hstack, &
    & ac_vstack, ac_stack_axis0, ac_stack_axis2
public :: ac_take, ac_put, ac_pad, ac_roll, ac_isnan, ac_isfinite, ac_nansum, ac_floor, ac_ceil, ac_nan, &
    & ac_random_integers, ac_shuffle, ac_choice_replace, ac_reduceat_add, ac_histogram_counts, &
    & ac_histogram_edges, ac_det
public :: ac_wall_time

interface ac_array
    module procedure ac_array_1d_real
    module procedure ac_array_2d_real
end interface

interface ac_asarray
    module procedure ac_asarray_scalar_real
    module procedure ac_asarray_1d_real
    module procedure ac_asarray_2d_real
    module procedure ac_asarray_scalar_int
    module procedure ac_asarray_1d_int
    module procedure ac_asarray_2d_int
    module procedure ac_asarray_1d_string
end interface

interface ac_astype_int
    module procedure ac_astype_int_1d_logical
    module procedure ac_astype_int_2d_logical
    module procedure ac_astype_int_1d_real
    module procedure ac_astype_int_2d_real
end interface

interface ac_astype_float
    module procedure ac_astype_float_1d_int
    module procedure ac_astype_float_2d_int
    module procedure ac_astype_float_1d_real
    module procedure ac_astype_float_2d_real
end interface

interface ac_copy
    module procedure ac_copy_1d_real
    module procedure ac_copy_2d_real
    module procedure ac_copy_1d_int
    module procedure ac_copy_2d_int
end interface

interface ac_sort
    module procedure ac_sort_1d_real
    module procedure ac_sort_1d_int
end interface

interface ac_unique
    module procedure ac_unique_1d_real
    module procedure ac_unique_1d_int
end interface

interface ac_bincount
    module procedure ac_bincount_1d_int
end interface

interface ac_searchsorted_left
    module procedure ac_searchsorted_left_1d_int
end interface

interface ac_searchsorted_right
    module procedure ac_searchsorted_right_1d_int
end interface

interface ac_column_stack
    module procedure ac_column_stack_2_real
    module procedure ac_column_stack_3_real
    module procedure ac_column_stack_4_real
    module procedure ac_column_stack_int_real3
end interface

interface ac_repeat
    module procedure ac_repeat_scalar_real
    module procedure ac_repeat_scalar_int
    module procedure ac_repeat_1d_real
    module procedure ac_repeat_1d_int
end interface

interface ac_take
    module procedure ac_take_1d_real
    module procedure ac_take_1d_int
end interface

interface ac_put
    module procedure ac_put_1d_real
    module procedure ac_put_1d_int
end interface

interface ac_pad
    module procedure ac_pad_1d_real
    module procedure ac_pad_1d_int
end interface

interface ac_roll
    module procedure ac_roll_1d_real
    module procedure ac_roll_1d_int
end interface

interface ac_floor
    module procedure ac_floor_1d_real
end interface

interface ac_ceil
    module procedure ac_ceil_1d_real
end interface

interface ac_isnan
    module procedure ac_isnan_scalar_real
    module procedure ac_isnan_1d_real
end interface

interface ac_isfinite
    module procedure ac_isfinite_scalar_real
    module procedure ac_isfinite_1d_real
end interface

interface ac_reduceat_add
    module procedure ac_reduceat_add_1d_real
    module procedure ac_reduceat_add_1d_int
end interface

interface ac_histogram_counts
    module procedure ac_histogram_counts_1d_real
end interface

interface ac_histogram_edges
    module procedure ac_histogram_edges_1d_real
end interface

interface ac_repeat_axis
    module procedure ac_repeat_axis_2d_real
    module procedure ac_repeat_axis_2d_int
end interface

interface ac_tile
    module procedure ac_tile_1d_int
    module procedure ac_tile_1d_real
    module procedure ac_tile_2d_int
    module procedure ac_tile_2d_real
end interface

interface ac_diag
    module procedure ac_diag_from_size_int
    module procedure ac_diag_from_size_real
    module procedure ac_diag_from_vec_real
    module procedure ac_diag_from_vec_int
    module procedure ac_diag_from_mat_real
    module procedure ac_diag_from_mat_int
end interface

interface ac_diag_k
    module procedure ac_diag_from_mat_real_k
end interface

interface ac_triu
    module procedure ac_triu_2d_real
end interface

interface ac_tril
    module procedure ac_tril_2d_real
end interface

interface ac_trace
    module procedure ac_trace_2d_real
end interface

interface ac_outer
    module procedure ac_outer_1d_real
end interface

interface ac_kron
    module procedure ac_kron_1d_real
end interface

interface ac_concatenate_axis0
    module procedure ac_concatenate_axis0_2d_real
    module procedure ac_concatenate_axis0_2d_int
end interface

interface ac_concatenate_axis1
    module procedure ac_concatenate_axis1_2d_real
    module procedure ac_concatenate_axis1_2d_int
end interface

interface ac_hstack
    module procedure ac_hstack_2d_real
    module procedure ac_hstack_2d_int
end interface

interface ac_vstack
    module procedure ac_vstack_2d_real
    module procedure ac_vstack_2d_int
end interface

interface ac_stack_axis0
    module procedure ac_stack_axis0_2d_real
    module procedure ac_stack_axis0_2d_int
end interface

interface ac_stack_axis2
    module procedure ac_stack_axis2_2d_real
    module procedure ac_stack_axis2_2d_int
end interface

interface ac_reverse
    module procedure ac_reverse_1d_real
    module procedure ac_reverse_1d_int
end interface

interface ac_r_concat
    module procedure ac_r_concat_array_scalar
    module procedure ac_r_concat_scalar_array
    module procedure ac_r_concat_array_array
end interface

interface ac_roots
    module procedure ac_roots_real
end interface

interface ac_zeros
    module procedure ac_zeros_1d
end interface

interface ac_full
    module procedure ac_full_1d
    module procedure ac_full_2d
    module procedure ac_full_1d_string
end interface

interface ac_clip
    module procedure ac_clip_scalar_real
    module procedure ac_clip_1d_real
end interface

interface ac_argmin
    module procedure ac_argmin_1d_real
    module procedure ac_argmin_2d_real
    module procedure ac_argmin_1d_int
    module procedure ac_argmin_2d_int
end interface

interface ac_argmax
    module procedure ac_argmax_1d_real
    module procedure ac_argmax_2d_real
    module procedure ac_argmax_1d_int
    module procedure ac_argmax_2d_int
end interface

interface ac_argmin_axis
    module procedure ac_argmin_axis_2d_real
    module procedure ac_argmin_axis_2d_int
end interface

interface ac_argmax_axis
    module procedure ac_argmax_axis_2d_real
    module procedure ac_argmax_axis_2d_int
end interface

interface ac_min_axis
    module procedure ac_min_axis_2d_real
    module procedure ac_min_axis_2d_int
end interface

interface ac_var
    module procedure ac_var_1d_real
end interface

interface ac_std
    module procedure ac_std_1d_real
end interface

interface ac_cumsum
    module procedure ac_cumsum_1d_real
    module procedure ac_cumsum_1d_int
end interface

interface ac_cumprod
    module procedure ac_cumprod_1d_real
    module procedure ac_cumprod_1d_int
end interface

interface ac_diff
    module procedure ac_diff_1d_real
end interface

interface ac_gradient
    module procedure ac_gradient_1d_real
end interface

interface ac_sign
    module procedure ac_sign_1d_real
end interface

interface ac_round
    module procedure ac_round_scalar
    module procedure ac_round_1d
end interface

interface ac_slice
    module procedure ac_slice_1d_real
    module procedure ac_slice_1d_int
end interface

interface ac_slice_step
    module procedure ac_slice_step_1d_real
    module procedure ac_slice_step_1d_int
end interface

interface ac_set_slice
    module procedure ac_set_slice_1d_real
end interface

interface ac_arange
    module procedure ac_arange_2
    module procedure ac_arange_3
end interface

interface ac_ndim
    module procedure ac_ndim_1d_real
    module procedure ac_ndim_2d_real
    module procedure ac_ndim_3d_real
    module procedure ac_ndim_1d_int
end interface

interface ac_any
    module procedure ac_any_scalar_logical
    module procedure ac_any_1d_logical
end interface

interface ac_shape_dim
    module procedure ac_shape_dim_1d_real
    module procedure ac_shape_dim_2d_real
    module procedure ac_shape_dim_3d_real
    module procedure ac_shape_dim_1d_int
    module procedure ac_shape_dim_2d_int
    module procedure ac_shape_dim_3d_int
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

interface ac_row_axis
    module procedure ac_row_axis_1d_real
    module procedure ac_row_axis_1d_int
end interface

interface ac_add_axis
    module procedure ac_add_axis_real
    module procedure ac_add_axis_int
end interface

interface ac_reshape
    module procedure ac_reshape_1d_to_1d_real
    module procedure ac_reshape_1d_to_1d_int
    module procedure ac_reshape_1d_to_1d_string
    module procedure ac_reshape_1d_to_2d_real
    module procedure ac_reshape_1d_to_2d_int
    module procedure ac_reshape_1d_to_3d_real
    module procedure ac_reshape_1d_to_3d_int
    module procedure ac_reshape_1d_to_2d_real_order
    module procedure ac_reshape_1d_to_2d_int_order
    module procedure ac_reshape_2d_to_1d_real
    module procedure ac_reshape_2d_to_1d_int
    module procedure ac_reshape_2d_to_2d_real
    module procedure ac_reshape_2d_to_2d_int
end interface

interface ac_mask
    module procedure ac_mask_1d_real
    module procedure ac_mask_2d_real
    module procedure ac_mask_1d_int
    module procedure ac_mask_2d_int
end interface

interface ac_pick2
    module procedure ac_pick2_2d_real
    module procedure ac_pick2_2d_int
end interface

interface ac_set_pick2
    module procedure ac_set_pick2_2d_real_scalar
    module procedure ac_set_pick2_2d_int_scalar
end interface

interface ac_slice2
    module procedure ac_slice2_2d_real
    module procedure ac_slice2_2d_int
end interface

interface ac_where_select
    module procedure ac_where_select_1d_real_real
    module procedure ac_where_select_1d_real_scalar
    module procedure ac_where_select_1d_scalar_real
    module procedure ac_where_select_1d_scalar_scalar_real
    module procedure ac_where_select_1d_int_int
    module procedure ac_where_select_1d_int_scalar
    module procedure ac_where_select_1d_scalar_int
    module procedure ac_where_select_1d_scalar_scalar_int
end interface

interface ac_savetxt
    module procedure ac_savetxt_1d_default
    module procedure ac_savetxt_1d
    module procedure ac_savetxt_2d_default
    module procedure ac_savetxt_2d
end interface

interface ac_atleast_2d
    module procedure ac_atleast_2d_real_1d
    module procedure ac_atleast_2d_real_2d
end interface

interface ac_argsort
    module procedure ac_argsort_real
    module procedure ac_argsort_int
end interface

interface ac_transpose
    module procedure ac_transpose_2d_real
end interface

interface ac_transpose_perm
    module procedure ac_transpose_perm_3d_real
    module procedure ac_transpose_perm_3d_int
end interface

interface ac_swapaxes
    module procedure ac_swapaxes_3d_real
    module procedure ac_swapaxes_3d_int
end interface

interface ac_matmul
    module procedure ac_matmul_2d_2d_real
    module procedure ac_matmul_2d_1d_real
    module procedure ac_matmul_1d_2d_real
    module procedure ac_matmul_1d_1d_real
end interface

interface ac_float
    module procedure ac_float_scalar
    module procedure ac_float_int_scalar
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

pure function ac_asarray_scalar_real(x) result(y)
real(dp), intent(in) :: x
real(dp), allocatable :: y(:)
allocate(y(1))
y(1) = x
end function ac_asarray_scalar_real

pure function ac_asarray_2d_real(x) result(y)
real(dp), intent(in) :: x(:, :)
real(dp), allocatable :: y(:, :)
y = x
end function ac_asarray_2d_real

pure function ac_asarray_1d_int(x) result(y)
integer, intent(in) :: x(:)
real(dp), allocatable :: y(:)
y = real(x, dp)
end function ac_asarray_1d_int

pure function ac_asarray_scalar_int(x) result(y)
integer, intent(in) :: x
real(dp), allocatable :: y(:)
allocate(y(1))
y(1) = real(x, dp)
end function ac_asarray_scalar_int

pure function ac_asarray_2d_int(x) result(y)
integer, intent(in) :: x(:, :)
real(dp), allocatable :: y(:, :)
y = real(x, dp)
end function ac_asarray_2d_int

pure function ac_asarray_1d_string(x) result(y)
character(len=*), intent(in) :: x(:)
character(len=:), allocatable :: y(:)
allocate(character(len=len(x)) :: y(size(x)))
y = x
end function ac_asarray_1d_string

pure function ac_astype_int_1d_logical(x) result(y)
logical, intent(in) :: x(:)
integer, allocatable :: y(:)
allocate(y(size(x)))
where (x)
    y = 1
elsewhere
    y = 0
end where
end function ac_astype_int_1d_logical

pure function ac_astype_int_2d_logical(x) result(y)
logical, intent(in) :: x(:, :)
integer, allocatable :: y(:, :)
allocate(y(size(x, 1), size(x, 2)))
where (x)
    y = 1
elsewhere
    y = 0
end where
end function ac_astype_int_2d_logical

pure function ac_astype_int_1d_real(x) result(y)
real(dp), intent(in) :: x(:)
integer, allocatable :: y(:)
allocate(y(size(x)))
y = int(x)
end function ac_astype_int_1d_real

pure function ac_astype_int_2d_real(x) result(y)
real(dp), intent(in) :: x(:, :)
integer, allocatable :: y(:, :)
allocate(y(size(x, 1), size(x, 2)))
y = int(x)
end function ac_astype_int_2d_real

pure function ac_astype_float_1d_int(x) result(y)
integer, intent(in) :: x(:)
real(dp), allocatable :: y(:)
allocate(y(size(x)))
y = real(x, dp)
end function ac_astype_float_1d_int

pure function ac_astype_float_2d_int(x) result(y)
integer, intent(in) :: x(:, :)
real(dp), allocatable :: y(:, :)
allocate(y(size(x, 1), size(x, 2)))
y = real(x, dp)
end function ac_astype_float_2d_int

pure function ac_astype_float_1d_real(x) result(y)
real(dp), intent(in) :: x(:)
real(dp), allocatable :: y(:)
y = x
end function ac_astype_float_1d_real

pure function ac_astype_float_2d_real(x) result(y)
real(dp), intent(in) :: x(:, :)
real(dp), allocatable :: y(:, :)
y = x
end function ac_astype_float_2d_real

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

pure function ac_copy_1d_int(x) result(y)
integer, intent(in) :: x(:)
integer, allocatable :: y(:)
y = x
end function ac_copy_1d_int

pure function ac_copy_2d_int(x) result(y)
integer, intent(in) :: x(:, :)
integer, allocatable :: y(:, :)
y = x
end function ac_copy_2d_int

pure function ac_reverse_1d_real(x) result(y)
real(dp), intent(in) :: x(:)
real(dp), allocatable :: y(:)
integer :: i, n
n = size(x)
allocate(y(n))
do i = 1, n
    y(i) = x(n - i + 1)
end do
end function ac_reverse_1d_real

pure function ac_reverse_1d_int(x) result(y)
integer, intent(in) :: x(:)
integer, allocatable :: y(:)
integer :: i, n
n = size(x)
allocate(y(n))
do i = 1, n
    y(i) = x(n - i + 1)
end do
end function ac_reverse_1d_int

pure function ac_take_1d_real(x, idx) result(y)
real(dp), intent(in) :: x(:)
integer, intent(in) :: idx(:)
real(dp), allocatable :: y(:)

allocate(y(size(idx)))
y = x(idx + 1)
end function ac_take_1d_real

pure function ac_take_1d_int(x, idx) result(y)
integer, intent(in) :: x(:)
integer, intent(in) :: idx(:)
integer, allocatable :: y(:)

allocate(y(size(idx)))
y = x(idx + 1)
end function ac_take_1d_int

subroutine ac_put_1d_real(x, idx, values)
real(dp), intent(inout) :: x(:)
integer, intent(in) :: idx(:)
real(dp), intent(in) :: values(:)

x(idx + 1) = values
end subroutine ac_put_1d_real

subroutine ac_put_1d_int(x, idx, values)
integer, intent(inout) :: x(:)
integer, intent(in) :: idx(:)
integer, intent(in) :: values(:)

x(idx + 1) = values
end subroutine ac_put_1d_int

pure function ac_pad_1d_real(x, left_pad, right_pad, constant_value) result(y)
real(dp), intent(in) :: x(:)
integer, intent(in) :: left_pad, right_pad
real(dp), intent(in) :: constant_value
real(dp), allocatable :: y(:)

allocate(y(left_pad + size(x) + right_pad))
y = constant_value
y(left_pad + 1:left_pad + size(x)) = x
end function ac_pad_1d_real

pure function ac_pad_1d_int(x, left_pad, right_pad, constant_value) result(y)
integer, intent(in) :: x(:)
integer, intent(in) :: left_pad, right_pad, constant_value
integer, allocatable :: y(:)

allocate(y(left_pad + size(x) + right_pad))
y = constant_value
y(left_pad + 1:left_pad + size(x)) = x
end function ac_pad_1d_int

pure function ac_roll_1d_real(x, shift) result(y)
real(dp), intent(in) :: x(:)
integer, intent(in) :: shift
real(dp), allocatable :: y(:)
integer :: n, s

n = size(x)
allocate(y(n))
if (n == 0) return
s = modulo(shift, n)
if (s == 0) then
    y = x
else
    y(1:s) = x(n - s + 1:n)
    y(s + 1:n) = x(1:n - s)
end if
end function ac_roll_1d_real

pure function ac_roll_1d_int(x, shift) result(y)
integer, intent(in) :: x(:)
integer, intent(in) :: shift
integer, allocatable :: y(:)
integer :: n, s

n = size(x)
allocate(y(n))
if (n == 0) return
s = modulo(shift, n)
if (s == 0) then
    y = x
else
    y(1:s) = x(n - s + 1:n)
    y(s + 1:n) = x(1:n - s)
end if
end function ac_roll_1d_int

pure function ac_r_concat_array_scalar(x, y) result(z)
real(dp), intent(in) :: x(:)
real(dp), intent(in) :: y
real(dp), allocatable :: z(:)
z = [x, y]
end function ac_r_concat_array_scalar

pure function ac_r_concat_scalar_array(x, y) result(z)
real(dp), intent(in) :: x
real(dp), intent(in) :: y(:)
real(dp), allocatable :: z(:)
z = [x, y]
end function ac_r_concat_scalar_array

pure function ac_r_concat_array_array(x, y) result(z)
real(dp), intent(in) :: x(:), y(:)
real(dp), allocatable :: z(:)
z = [x, y]
end function ac_r_concat_array_array

pure function ac_roots_real(coeffs) result(roots)
! Find roots of a real polynomial using Durand-Kerner iteration.
real(dp), intent(in) :: coeffs(:)
complex(dp), allocatable :: roots(:)
real(dp), allocatable :: work_coeffs(:)
complex(dp), allocatable :: current(:), next_roots(:)
real(dp) :: radius, tol, theta
integer :: n, i, j, iter, max_iter, first_nz
complex(dp) :: denom, delta

first_nz = 0
do i = 1, size(coeffs)
    if (abs(coeffs(i)) > 0.0_dp) then
        first_nz = i
        exit
    end if
end do
if (first_nz == 0) then
    allocate(roots(0))
    return
end if

work_coeffs = coeffs(first_nz:)
n = size(work_coeffs) - 1
if (n < 1) then
    allocate(roots(0))
    return
end if
if (n == 1) then
    allocate(roots(1))
    roots(1) = cmplx(-work_coeffs(2) / work_coeffs(1), 0.0_dp, kind=dp)
    return
end if

radius = 1.0_dp + maxval(abs(work_coeffs(2:))) / abs(work_coeffs(1))
tol = 1.0e-12_dp
max_iter = 200
allocate(current(n), next_roots(n), roots(n))
do i = 1, n
    theta = 2.0_dp * acos(-1.0_dp) * real(i - 1, dp) / real(n, dp)
    current(i) = radius * cmplx(cos(theta), sin(theta), kind=dp)
end do

do iter = 1, max_iter
    do i = 1, n
        denom = cmplx(1.0_dp, 0.0_dp, kind=dp)
        do j = 1, n
            if (j /= i) denom = denom * (current(i) - current(j))
        end do
        if (abs(denom) <= tol) denom = cmplx(tol, 0.0_dp, kind=dp)
        delta = ac_poly_eval_real_complex(work_coeffs, current(i)) / denom
        next_roots(i) = current(i) - delta
    end do
    if (maxval(abs(next_roots - current)) <= tol) exit
    current = next_roots
end do
roots = next_roots
end function ac_roots_real

pure function ac_poly_eval_real_complex(coeffs, z) result(value)
! Evaluate a real-coefficient polynomial at the complex point z.
real(dp), intent(in) :: coeffs(:)
complex(dp), intent(in) :: z
complex(dp) :: value
integer :: i

value = cmplx(coeffs(1), 0.0_dp, kind=dp)
do i = 2, size(coeffs)
    value = (value * z) + cmplx(coeffs(i), 0.0_dp, kind=dp)
end do
end function ac_poly_eval_real_complex

function ac_normal_vec(state, loc, scale, n) result(x)
real(dp), intent(in) :: loc, scale
integer, intent(in) :: n
type(ac_random_state), intent(in) :: state
real(dp), allocatable :: x(:)
type(ac_random_state) :: work_state
integer :: i

work_state = state
allocate(x(n))
do i = 1, n
    x(i) = ac_gauss(work_state, loc, scale)
end do
end function ac_normal_vec

function ac_uniform_vec(state, low, high, n) result(x)
real(dp), intent(in) :: low, high
integer, intent(in) :: n
type(ac_random_state), intent(in) :: state
real(dp), allocatable :: x(:)
integer :: i
integer :: nseed
integer, allocatable :: seed(:)

call random_seed(size=nseed)
allocate(seed(nseed))
seed = state%seed
call random_seed(put=seed)
allocate(x(n))
do i = 1, n
    call random_number(x(i))
end do
x = low + (high - low) * x
end function ac_uniform_vec

function ac_random_integers(state, low, high, n) result(x)
type(ac_random_state), intent(in) :: state
integer, intent(in) :: low, high, n
integer, allocatable :: x(:)
real(dp) :: u
integer :: i
integer :: nseed
integer, allocatable :: seed(:)

call random_seed(size=nseed)
allocate(seed(nseed))
seed = state%seed
call random_seed(put=seed)
allocate(x(n))
do i = 1, n
    call random_number(u)
    x(i) = low + min(high - low - 1, int(u * max(1, high - low)))
end do
end function ac_random_integers

subroutine ac_shuffle(state, x)
type(ac_random_state), intent(in) :: state
integer, intent(inout) :: x(:)
real(dp) :: u
integer :: i, j, temp
integer :: nseed
integer, allocatable :: seed(:)

call random_seed(size=nseed)
allocate(seed(nseed))
seed = state%seed
call random_seed(put=seed)
do i = size(x), 2, -1
    call random_number(u)
    j = min(i, int(u * i) + 1)
    temp = x(i)
    x(i) = x(j)
    x(j) = temp
end do
end subroutine ac_shuffle

pure function ac_zeros_1d(n) result(x)
integer, intent(in) :: n
real(dp), allocatable :: x(:)
allocate(x(n))
x = 0.0_dp
end function ac_zeros_1d

pure function ac_zeros2(n, m) result(x)
integer, intent(in) :: n, m
real(dp), allocatable :: x(:, :)
allocate(x(n, m))
x = 0.0_dp
end function ac_zeros2

pure function ac_dot(x, y) result(value)
real(dp), intent(in) :: x(:), y(:)
real(dp) :: value
value = sum(x * y)
end function ac_dot

pure function ac_mean(x) result(value)
real(dp), intent(in) :: x(:)
real(dp) :: value
value = sum(x) / real(size(x), dp)
end function ac_mean

pure function ac_slice_1d_real(x, start_index, stop_index) result(y)
real(dp), intent(in) :: x(:)
integer, intent(in) :: start_index, stop_index
real(dp), allocatable :: y(:)
integer :: lo, hi

lo = start_index
if (lo < 0) lo = size(x) + lo
lo = max(0, lo) + 1
hi = stop_index
if (hi < 0) hi = size(x) + hi
hi = min(size(x), hi)
if (hi < lo) then
    y = [real(dp) :: ]
else
    y = x(lo:hi)
end if
end function ac_slice_1d_real

pure function ac_slice_1d_int(x, start_index, stop_index) result(y)
integer, intent(in) :: x(:)
integer, intent(in) :: start_index, stop_index
integer, allocatable :: y(:)
integer :: lo, hi

lo = start_index
if (lo < 0) lo = size(x) + lo
lo = max(0, lo) + 1
hi = stop_index
if (hi < 0) hi = size(x) + hi
hi = min(size(x), hi)
if (hi < lo) then
    y = [integer :: ]
else
    y = x(lo:hi)
end if
end function ac_slice_1d_int

pure function ac_slice_step_1d_real(x, start_index, stop_index, step_value) result(y)
real(dp), intent(in) :: x(:)
integer, intent(in) :: start_index, stop_index, step_value
real(dp), allocatable :: y(:)
integer :: lo, hi

lo = start_index
if (lo < 0) lo = size(x) + lo
hi = stop_index
if (hi < 0) hi = size(x) + hi
if (step_value > 0) then
    lo = max(0, lo) + 1
    hi = min(size(x), hi)
    if (hi < lo) then
        y = [real(dp) :: ]
    else
        y = x(lo:hi:step_value)
    end if
else
    lo = min(size(x) - 1, lo) + 1
    hi = max(-1, hi) + 1
    if (lo < 1 .or. lo < hi) then
        y = [real(dp) :: ]
    else
        y = x(lo:hi:step_value)
    end if
end if
end function ac_slice_step_1d_real

pure function ac_slice_step_1d_int(x, start_index, stop_index, step_value) result(y)
integer, intent(in) :: x(:)
integer, intent(in) :: start_index, stop_index, step_value
integer, allocatable :: y(:)
integer :: lo, hi

lo = start_index
if (lo < 0) lo = size(x) + lo
hi = stop_index
if (hi < 0) hi = size(x) + hi
if (step_value > 0) then
    lo = max(0, lo) + 1
    hi = min(size(x), hi)
    if (hi < lo) then
        y = [integer :: ]
    else
        y = x(lo:hi:step_value)
    end if
else
    lo = min(size(x) - 1, lo) + 1
    hi = max(-1, hi) + 1
    if (lo < 1 .or. lo < hi) then
        y = [integer :: ]
    else
        y = x(lo:hi:step_value)
    end if
end if
end function ac_slice_step_1d_int

pure subroutine ac_set_slice_1d_real(x, start_index, stop_index, values)
real(dp), intent(inout) :: x(:)
integer, intent(in) :: start_index, stop_index
real(dp), intent(in) :: values(:)
integer :: lo, hi

lo = start_index
if (lo < 0) lo = size(x) + lo
lo = max(0, lo) + 1
hi = stop_index
if (hi < 0) hi = size(x) + hi
hi = min(size(x), hi)
if (hi >= lo) x(lo:hi) = values
end subroutine ac_set_slice_1d_real

pure subroutine ac_fill_slice(x, start_index, stop_index, value)
real(dp), intent(inout) :: x(:)
integer, intent(in) :: start_index, stop_index
real(dp), intent(in) :: value
integer :: lo, hi

lo = start_index
if (lo < 0) lo = size(x) + lo
lo = max(0, lo) + 1
hi = stop_index
if (hi < 0) hi = size(x) + hi
hi = min(size(x), hi)
if (hi >= lo) x(lo:hi) = value
end subroutine ac_fill_slice

pure function ac_arange_2(start_value, stop_value) result(x)
integer, intent(in) :: start_value, stop_value
real(dp), allocatable :: x(:)
integer :: n, i

n = max(0, stop_value - start_value)
allocate(x(n))
do i = 1, n
    x(i) = real(start_value + i - 1, dp)
end do
end function ac_arange_2

pure function ac_arange_3(start_value, stop_value, step_value) result(x)
integer, intent(in) :: start_value, stop_value, step_value
real(dp), allocatable :: x(:)
integer :: n, i, value

if (step_value == 0) then
    allocate(x(0))
    return
end if
if ((step_value > 0 .and. start_value >= stop_value) .or. (step_value < 0 .and. start_value <= stop_value)) then
    allocate(x(0))
    return
end if
n = 0
value = start_value
do while ((step_value > 0 .and. value < stop_value) .or. (step_value < 0 .and. value > stop_value))
    n = n + 1
    value = value + step_value
end do
allocate(x(n))
value = start_value
do i = 1, n
    x(i) = real(value, dp)
    value = value + step_value
end do
end function ac_arange_3

pure function ac_linspace(start_value, stop_value, num_points) result(x)
real(dp), intent(in) :: start_value, stop_value
integer, intent(in) :: num_points
real(dp), allocatable :: x(:)
integer :: i

if (num_points <= 0) then
    allocate(x(0))
else if (num_points == 1) then
    allocate(x(1))
    x(1) = start_value
else
    allocate(x(num_points))
    do i = 1, num_points
        x(i) = start_value + (stop_value - start_value) * real(i - 1, dp) / real(num_points - 1, dp)
    end do
    x(1) = start_value
    x(num_points) = stop_value
end if
end function ac_linspace

pure function ac_arange_int(start_value, stop_value) result(x)
integer, intent(in) :: start_value, stop_value
integer, allocatable :: x(:)
integer :: n, i

n = max(0, stop_value - start_value)
allocate(x(n))
do i = 1, n
    x(i) = start_value + i - 1
end do
end function ac_arange_int

pure function ac_column_stack_4_real(a, b, c, d) result(x)
real(dp), intent(in) :: a(:), b(:), c(:), d(:)
real(dp), allocatable :: x(:, :)
integer :: n

n = size(a)
allocate(x(n, 4))
x(:, 1) = a
x(:, 2) = b
x(:, 3) = c
x(:, 4) = d
end function ac_column_stack_4_real

pure function ac_column_stack_2_real(a, b) result(x)
real(dp), intent(in) :: a(:), b(:)
real(dp), allocatable :: x(:, :)
integer :: n
n = size(a)
allocate(x(n, 2))
x(:, 1) = a
x(:, 2) = b
end function ac_column_stack_2_real

pure function ac_column_stack_3_real(a, b, c) result(x)
real(dp), intent(in) :: a(:), b(:), c(:)
real(dp), allocatable :: x(:, :)
integer :: n

n = size(a)
allocate(x(n, 3))
x(:, 1) = a
x(:, 2) = b
x(:, 3) = c
end function ac_column_stack_3_real

pure function ac_column_stack_int_real3(a, b, c, d) result(x)
integer, intent(in) :: a(:)
real(dp), intent(in) :: b(:), c(:), d(:)
real(dp), allocatable :: x(:, :)
integer :: n

n = size(a)
allocate(x(n, 4))
x(:, 1) = real(a, dp)
x(:, 2) = b
x(:, 3) = c
x(:, 4) = d
end function ac_column_stack_int_real3

pure function ac_round_scalar(x, decimals) result(y)
real(dp), intent(in) :: x
integer, intent(in) :: decimals
real(dp) :: y
real(dp) :: scale

scale = 10.0_dp ** decimals
y = anint(x * scale) / scale
end function ac_round_scalar

pure function ac_floor_1d_real(x) result(y)
real(dp), intent(in) :: x(:)
real(dp), allocatable :: y(:)

allocate(y(size(x)))
y = floor(x)
end function ac_floor_1d_real

pure function ac_ceil_1d_real(x) result(y)
real(dp), intent(in) :: x(:)
real(dp), allocatable :: y(:)

allocate(y(size(x)))
y = ceiling(x)
end function ac_ceil_1d_real

pure function ac_nan() result(value)
real(dp) :: value

value = ieee_value(0.0_dp, ieee_quiet_nan)
end function ac_nan

pure function ac_isnan_1d_real(x) result(mask)
real(dp), intent(in) :: x(:)
logical, allocatable :: mask(:)

allocate(mask(size(x)))
mask = ieee_is_nan(x)
end function ac_isnan_1d_real

pure elemental function ac_isnan_scalar_real(x) result(mask)
real(dp), intent(in) :: x
logical :: mask

mask = ieee_is_nan(x)
end function ac_isnan_scalar_real

pure elemental function ac_isfinite_scalar_real(x) result(mask)
real(dp), intent(in) :: x
logical :: mask

mask = ieee_is_finite(x)
end function ac_isfinite_scalar_real

pure function ac_isfinite_1d_real(x) result(mask)
real(dp), intent(in) :: x(:)
logical, allocatable :: mask(:)

allocate(mask(size(x)))
mask = ieee_is_finite(x)
end function ac_isfinite_1d_real

pure function ac_nansum(x) result(value)
real(dp), intent(in) :: x(:)
real(dp) :: value

value = sum(merge(x, 0.0_dp, .not. ieee_is_nan(x)))
end function ac_nansum

pure function ac_round_1d(x, decimals) result(y)
real(dp), intent(in) :: x(:)
integer, intent(in) :: decimals
real(dp), allocatable :: y(:)
real(dp) :: scale

scale = 10.0_dp ** decimals
y = anint(x * scale) / scale
end function ac_round_1d

pure real(dp) function ac_clip_scalar_real(x, lower, upper)
real(dp), intent(in) :: x, lower, upper
ac_clip_scalar_real = min(max(x, lower), upper)
end function ac_clip_scalar_real

pure function ac_clip_1d_real(x, lower, upper) result(y)
real(dp), intent(in) :: x(:)
real(dp), intent(in) :: lower, upper
real(dp), allocatable :: y(:)
y = min(max(x, lower), upper)
end function ac_clip_1d_real

pure function ac_sign_1d_real(x) result(y)
real(dp), intent(in) :: x(:)
real(dp), allocatable :: y(:)
y = sign(1.0_dp, x)
where (x == 0.0_dp)
    y = 0.0_dp
end where
end function ac_sign_1d_real

pure logical function ac_any_scalar_logical(x)
logical, intent(in) :: x
ac_any_scalar_logical = x
end function ac_any_scalar_logical

pure logical function ac_any_1d_logical(x)
logical, intent(in) :: x(:)
ac_any_1d_logical = any(x)
end function ac_any_1d_logical

pure real(dp) function ac_norm(x)
real(dp), intent(in) :: x(:)
ac_norm = sqrt(sum(x**2))
end function ac_norm

pure function ac_solve_linear(a, b) result(x)
real(dp), intent(in) :: a(:, :), b(:)
real(dp), allocatable :: x(:)
x = matmul(ac_inv(a), b)
end function ac_solve_linear

pure function ac_solve_linear_fallback(a, b) result(x)
real(dp), intent(in) :: a(:, :), b(:)
real(dp), allocatable :: x(:)
real(dp), allocatable :: ata(:, :), atb(:)

ata = matmul(transpose(a), a)
atb = matmul(transpose(a), b)
x = ac_solve_linear(ata, atb)
end function ac_solve_linear_fallback

pure function ac_item2(x, i, j) result(value)
real(dp), intent(in) :: x(:, :)
integer, intent(in) :: i, j
real(dp) :: value
value = x(i + 1, j + 1)
end function ac_item2

pure subroutine ac_set_item2(x, i, j, value)
real(dp), intent(inout) :: x(:, :)
integer, intent(in) :: i, j
real(dp), intent(in) :: value
x(i + 1, j + 1) = value
end subroutine ac_set_item2

pure integer function ac_ndim_1d_real(x)
real(dp), intent(in) :: x(:)
ac_ndim_1d_real = 1
end function ac_ndim_1d_real

pure integer function ac_ndim_2d_real(x)
real(dp), intent(in) :: x(:, :)
ac_ndim_2d_real = 2
end function ac_ndim_2d_real

pure integer function ac_ndim_3d_real(x)
real(dp), intent(in) :: x(:, :, :)
ac_ndim_3d_real = 3
end function ac_ndim_3d_real

pure integer function ac_ndim_1d_int(x)
integer, intent(in) :: x(:)
ac_ndim_1d_int = 1
end function ac_ndim_1d_int

pure integer function ac_shape_dim_1d_real(x, dim_index)
real(dp), intent(in) :: x(:)
integer, intent(in) :: dim_index
if (dim_index == 1) ac_shape_dim_1d_real = size(x, 1)
if (dim_index /= 1) ac_shape_dim_1d_real = 1
end function ac_shape_dim_1d_real

pure integer function ac_shape_dim_2d_real(x, dim_index)
real(dp), intent(in) :: x(:, :)
integer, intent(in) :: dim_index
ac_shape_dim_2d_real = size(x, dim_index)
end function ac_shape_dim_2d_real

pure integer function ac_shape_dim_3d_real(x, dim_index)
real(dp), intent(in) :: x(:, :, :)
integer, intent(in) :: dim_index
ac_shape_dim_3d_real = size(x, dim_index)
end function ac_shape_dim_3d_real

pure integer function ac_shape_dim_1d_int(x, dim_index)
integer, intent(in) :: x(:)
integer, intent(in) :: dim_index
if (dim_index == 1) ac_shape_dim_1d_int = size(x, 1)
if (dim_index /= 1) ac_shape_dim_1d_int = 1
end function ac_shape_dim_1d_int

pure integer function ac_shape_dim_2d_int(x, dim_index)
integer, intent(in) :: x(:, :)
integer, intent(in) :: dim_index
if (dim_index == 1) ac_shape_dim_2d_int = size(x, 1)
if (dim_index /= 1) ac_shape_dim_2d_int = size(x, 2)
end function ac_shape_dim_2d_int

pure integer function ac_shape_dim_3d_int(x, dim_index)
integer, intent(in) :: x(:, :, :)
integer, intent(in) :: dim_index
select case (dim_index)
case (1)
    ac_shape_dim_3d_int = size(x, 1)
case (2)
    ac_shape_dim_3d_int = size(x, 2)
case default
    ac_shape_dim_3d_int = size(x, 3)
end select
end function ac_shape_dim_3d_int

pure function ac_empty2(n, m) result(x)
integer, intent(in) :: n, m
real(dp), allocatable :: x(:, :)
allocate(x(n, m))
end function ac_empty2

pure function ac_empty(n) result(x)
integer, intent(in) :: n
real(dp), allocatable :: x(:)
allocate(x(n))
end function ac_empty

pure function ac_full_1d(n, value) result(x)
integer, intent(in) :: n
real(dp), intent(in) :: value
real(dp), allocatable :: x(:)
allocate(x(n))
x = value
end function ac_full_1d

pure function ac_full_2d(n, m, value) result(x)
    integer, intent(in) :: n, m
    real(dp), intent(in) :: value
    real(dp), allocatable :: x(:, :)
    allocate(x(n, m))
    x = value
end function ac_full_2d

pure function ac_full_1d_string(n, value) result(x)
    integer, intent(in) :: n
    character(len=*), intent(in) :: value
    character(len=:), allocatable :: x(:)
    allocate(character(len=max(len(value), 64)) :: x(n))
    x = value
end function ac_full_1d_string

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

function ac_choice_replace(state, population, nsamp) result(z)
type(ac_random_state), intent(inout) :: state
integer, intent(in) :: population(:)
integer, intent(in) :: nsamp
integer, allocatable :: z(:)
real(dp) :: u
integer :: i, pick

allocate(z(nsamp))
do i = 1, nsamp
    call random_number(u)
    pick = min(size(population), int(u * size(population)) + 1)
    z(i) = population(pick)
end do
end function ac_choice_replace

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

pure function ac_mask_1d_real(x, mask) result(y)
real(dp), intent(in) :: x(:)
logical, intent(in) :: mask(:)
real(dp), allocatable :: y(:)
integer :: n

n = count(mask)
allocate(y(n))
y = pack(x, mask)
end function ac_mask_1d_real

pure function ac_mask_2d_real(x, mask) result(y)
real(dp), intent(in) :: x(:, :)
logical, intent(in) :: mask(:, :)
real(dp), allocatable :: y(:)
integer :: n

n = count(mask)
allocate(y(n))
y = pack(x, mask)
end function ac_mask_2d_real

pure function ac_mask_1d_int(x, mask) result(y)
integer, intent(in) :: x(:)
logical, intent(in) :: mask(:)
integer, allocatable :: y(:)
integer :: n

n = count(mask)
allocate(y(n))
y = pack(x, mask)
end function ac_mask_1d_int

pure function ac_mask_2d_int(x, mask) result(y)
integer, intent(in) :: x(:, :)
logical, intent(in) :: mask(:, :)
integer, allocatable :: y(:)
integer :: n

n = count(mask)
allocate(y(n))
y = pack(x, mask)
end function ac_mask_2d_int

pure function ac_pick2_2d_real(x, rows, cols) result(y)
real(dp), intent(in) :: x(:, :)
integer, intent(in) :: rows(:), cols(:)
real(dp), allocatable :: y(:)
integer :: i

if (size(rows) /= size(cols)) error stop "ac_pick2 size mismatch"
allocate(y(size(rows)))
do i = 1, size(rows)
    y(i) = x(rows(i) + 1, cols(i) + 1)
end do
end function ac_pick2_2d_real

pure function ac_pick2_2d_int(x, rows, cols) result(y)
integer, intent(in) :: x(:, :)
integer, intent(in) :: rows(:), cols(:)
integer, allocatable :: y(:)
integer :: i

if (size(rows) /= size(cols)) error stop "ac_pick2 size mismatch"
allocate(y(size(rows)))
do i = 1, size(rows)
    y(i) = x(rows(i) + 1, cols(i) + 1)
end do
end function ac_pick2_2d_int

pure subroutine ac_set_pick2_2d_real_scalar(x, rows, cols, value)
real(dp), intent(inout) :: x(:, :)
integer, intent(in) :: rows(:), cols(:)
real(dp), intent(in) :: value
integer :: i

if (size(rows) /= size(cols)) error stop "ac_set_pick2 size mismatch"
do i = 1, size(rows)
    x(rows(i) + 1, cols(i) + 1) = value
end do
end subroutine ac_set_pick2_2d_real_scalar

pure subroutine ac_set_pick2_2d_int_scalar(x, rows, cols, value)
integer, intent(inout) :: x(:, :)
integer, intent(in) :: rows(:), cols(:)
integer, intent(in) :: value
integer :: i

if (size(rows) /= size(cols)) error stop "ac_set_pick2 size mismatch"
do i = 1, size(rows)
    x(rows(i) + 1, cols(i) + 1) = value
end do
end subroutine ac_set_pick2_2d_int_scalar

pure function ac_slice2_2d_real(x, r0, r1, rs, c0, c1, cs) result(y)
real(dp), intent(in) :: x(:, :)
integer, intent(in) :: r0, r1, rs, c0, c1, cs
real(dp), allocatable :: y(:, :)

allocate(y(size(x(r0 + 1:r1:rs, c0 + 1:c1:cs), 1), size(x(r0 + 1:r1:rs, c0 + 1:c1:cs), 2)))
y = x(r0 + 1:r1:rs, c0 + 1:c1:cs)
end function ac_slice2_2d_real

pure function ac_slice2_2d_int(x, r0, r1, rs, c0, c1, cs) result(y)
integer, intent(in) :: x(:, :)
integer, intent(in) :: r0, r1, rs, c0, c1, cs
integer, allocatable :: y(:, :)

allocate(y(size(x(r0 + 1:r1:rs, c0 + 1:c1:cs), 1), size(x(r0 + 1:r1:rs, c0 + 1:c1:cs), 2)))
y = x(r0 + 1:r1:rs, c0 + 1:c1:cs)
end function ac_slice2_2d_int

pure function ac_where_select_1d_real_real(mask, x_true, x_false) result(y)
logical, intent(in) :: mask(:)
real(dp), intent(in) :: x_true(:), x_false(:)
real(dp), allocatable :: y(:)
allocate(y(size(mask)))
where (mask)
    y = x_true
elsewhere
    y = x_false
end where
end function ac_where_select_1d_real_real

pure function ac_where_select_1d_real_scalar(mask, x_true, x_false) result(y)
logical, intent(in) :: mask(:)
real(dp), intent(in) :: x_true(:)
real(dp), intent(in) :: x_false
real(dp), allocatable :: y(:)
allocate(y(size(mask)))
where (mask)
    y = x_true
elsewhere
    y = x_false
end where
end function ac_where_select_1d_real_scalar

pure function ac_where_select_1d_scalar_real(mask, x_true, x_false) result(y)
logical, intent(in) :: mask(:)
real(dp), intent(in) :: x_true
real(dp), intent(in) :: x_false(:)
real(dp), allocatable :: y(:)
allocate(y(size(mask)))
where (mask)
    y = x_true
elsewhere
    y = x_false
end where
end function ac_where_select_1d_scalar_real

pure function ac_where_select_1d_scalar_scalar_real(mask, x_true, x_false) result(y)
logical, intent(in) :: mask(:)
real(dp), intent(in) :: x_true, x_false
real(dp), allocatable :: y(:)
allocate(y(size(mask)))
where (mask)
    y = x_true
elsewhere
    y = x_false
end where
end function ac_where_select_1d_scalar_scalar_real

pure function ac_where_select_1d_int_int(mask, x_true, x_false) result(y)
logical, intent(in) :: mask(:)
integer, intent(in) :: x_true(:), x_false(:)
integer, allocatable :: y(:)
allocate(y(size(mask)))
where (mask)
    y = x_true
elsewhere
    y = x_false
end where
end function ac_where_select_1d_int_int

pure function ac_where_select_1d_int_scalar(mask, x_true, x_false) result(y)
logical, intent(in) :: mask(:)
integer, intent(in) :: x_true(:)
integer, intent(in) :: x_false
integer, allocatable :: y(:)
allocate(y(size(mask)))
where (mask)
    y = x_true
elsewhere
    y = x_false
end where
end function ac_where_select_1d_int_scalar

pure function ac_where_select_1d_scalar_int(mask, x_true, x_false) result(y)
logical, intent(in) :: mask(:)
integer, intent(in) :: x_true
integer, intent(in) :: x_false(:)
integer, allocatable :: y(:)
allocate(y(size(mask)))
where (mask)
    y = x_true
elsewhere
    y = x_false
end where
end function ac_where_select_1d_scalar_int

pure function ac_where_select_1d_scalar_scalar_int(mask, x_true, x_false) result(y)
logical, intent(in) :: mask(:)
integer, intent(in) :: x_true, x_false
integer, allocatable :: y(:)
allocate(y(size(mask)))
where (mask)
    y = x_true
elsewhere
    y = x_false
end where
end function ac_where_select_1d_scalar_scalar_int

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

pure function ac_row_axis_1d_real(x) result(y)
real(dp), intent(in) :: x(:)
real(dp), allocatable :: y(:, :)
allocate(y(1, size(x)))
y(1, :) = x
end function ac_row_axis_1d_real

pure function ac_row_axis_1d_int(x) result(y)
integer, intent(in) :: x(:)
integer, allocatable :: y(:, :)
allocate(y(1, size(x)))
y(1, :) = x
end function ac_row_axis_1d_int

pure subroutine ac_set_rows(x, idx, values)
real(dp), intent(inout) :: x(:, :)
integer, intent(in) :: idx(:)
real(dp), intent(in) :: values(:, :)
integer :: i

do i = 1, size(idx)
    x(idx(i) + 1, :) = values(i, :)
end do
end subroutine ac_set_rows

pure subroutine ac_set_row(x, row_index, values)
real(dp), intent(inout) :: x(:, :)
integer, intent(in) :: row_index
real(dp), intent(in) :: values(:)

x(row_index + 1, :) = values
end subroutine ac_set_row

pure subroutine ac_set_column(x, col_index, values)
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

pure function ac_reshape_1d_to_1d_real(x, n) result(y)
real(dp), intent(in) :: x(:)
integer, intent(in) :: n
real(dp), allocatable :: y(:)

if (n /= -1 .and. n /= size(x)) error stop "ac_reshape 1d size mismatch"
allocate(y(size(x)))
y = x
end function ac_reshape_1d_to_1d_real

pure function ac_reshape_1d_to_2d_int(x, n, m) result(y)
integer, intent(in) :: x(:)
integer, intent(in) :: n, m
integer, allocatable :: y(:, :)
integer :: i, j

allocate(y(n, m))
do j = 1, m
    do i = 1, n
        y(i, j) = x((i - 1) * m + j)
    end do
end do
end function ac_reshape_1d_to_2d_int

pure function ac_reshape_1d_to_1d_int(x, n) result(y)
integer, intent(in) :: x(:)
integer, intent(in) :: n
integer, allocatable :: y(:)

if (n /= -1 .and. n /= size(x)) error stop "ac_reshape 1d size mismatch"
allocate(y(size(x)))
y = x
end function ac_reshape_1d_to_1d_int

pure function ac_reshape_1d_to_1d_string(x, n) result(y)
character(len=*), intent(in) :: x(:)
integer, intent(in) :: n
character(len=:), allocatable :: y(:)

if (n /= -1 .and. n /= size(x)) error stop "ac_reshape 1d size mismatch"
allocate(character(len=len(x)) :: y(size(x)))
y = x
end function ac_reshape_1d_to_1d_string

pure function ac_reshape_1d_to_3d_real(x, n, m, p) result(y)
real(dp), intent(in) :: x(:)
integer, intent(in) :: n, m, p
real(dp), allocatable :: y(:, :, :)
integer :: i, j, k, idx

allocate(y(n, m, p))
idx = 0
do i = 1, n
    do j = 1, m
        do k = 1, p
            idx = idx + 1
            y(i, j, k) = x(idx)
end do
end do
end do
end function ac_reshape_1d_to_3d_real

pure function ac_reshape_1d_to_3d_int(x, n, m, p) result(y)
integer, intent(in) :: x(:)
integer, intent(in) :: n, m, p
integer, allocatable :: y(:, :, :)
integer :: i, j, k, idx

allocate(y(n, m, p))
idx = 0
do i = 1, n
    do j = 1, m
        do k = 1, p
            idx = idx + 1
            y(i, j, k) = x(idx)
        end do
    end do
end do
end function ac_reshape_1d_to_3d_int

pure function ac_reshape_1d_to_2d_real_order(x, n, m, order) result(y)
real(dp), intent(in) :: x(:)
integer, intent(in) :: n, m
character(len=*), intent(in) :: order
real(dp), allocatable :: y(:, :)
integer :: i, j

allocate(y(n, m))
if (order == "F") then
    y = reshape(x, [n, m])
else
    do j = 1, m
        do i = 1, n
            y(i, j) = x((i - 1) * m + j)
        end do
    end do
end if
end function ac_reshape_1d_to_2d_real_order

pure function ac_reshape_1d_to_2d_int_order(x, n, m, order) result(y)
integer, intent(in) :: x(:)
integer, intent(in) :: n, m
character(len=*), intent(in) :: order
integer, allocatable :: y(:, :)
integer :: i, j

allocate(y(n, m))
if (order == "F") then
    y = reshape(x, [n, m])
else
    do j = 1, m
        do i = 1, n
            y(i, j) = x((i - 1) * m + j)
        end do
    end do
end if
end function ac_reshape_1d_to_2d_int_order

pure function ac_reshape_2d_to_1d_real(x, n) result(y)
real(dp), intent(in) :: x(:, :)
integer, intent(in) :: n
real(dp), allocatable :: y(:)
integer :: i, j, k, n_out

n_out = merge(size(x), n, n == -1)
allocate(y(n_out))
k = 0
do i = 1, size(x, 1)
    do j = 1, size(x, 2)
        k = k + 1
        if (k <= n_out) y(k) = x(i, j)
    end do
end do
end function ac_reshape_2d_to_1d_real

pure function ac_reshape_2d_to_1d_int(x, n) result(y)
integer, intent(in) :: x(:, :)
integer, intent(in) :: n
integer, allocatable :: y(:)
integer :: i, j, k, n_out

n_out = merge(size(x), n, n == -1)
allocate(y(n_out))
k = 0
do i = 1, size(x, 1)
    do j = 1, size(x, 2)
        k = k + 1
        if (k <= n_out) y(k) = x(i, j)
    end do
end do
end function ac_reshape_2d_to_1d_int

pure function ac_reshape_2d_to_2d_real(x, n, m) result(y)
real(dp), intent(in) :: x(:, :)
integer, intent(in) :: n, m
real(dp), allocatable :: y(:, :)
integer :: i, j, k, n_out, m_out

if (n == -1 .and. m == -1) error stop "ac_reshape 2d target dims ambiguous"
if (n == -1) then
    if (m <= 0 .or. mod(size(x), m) /= 0) error stop "ac_reshape 2d size mismatch"
    m_out = m
    n_out = size(x) / m_out
elseif (m == -1) then
    if (n <= 0 .or. mod(size(x), n) /= 0) error stop "ac_reshape 2d size mismatch"
    n_out = n
    m_out = size(x) / n_out
else
    n_out = n
    m_out = m
    if (n_out * m_out /= size(x)) error stop "ac_reshape 2d size mismatch"
end if

allocate(y(n_out, m_out))
k = 0
do i = 1, size(x, 1)
    do j = 1, size(x, 2)
        k = k + 1
        y((k - 1) / m_out + 1, mod(k - 1, m_out) + 1) = x(i, j)
    end do
end do
end function ac_reshape_2d_to_2d_real

pure function ac_reshape_2d_to_2d_int(x, n, m) result(y)
integer, intent(in) :: x(:, :)
integer, intent(in) :: n, m
integer, allocatable :: y(:, :)
integer :: i, j, k, n_out, m_out

if (n == -1 .and. m == -1) error stop "ac_reshape 2d target dims ambiguous"
if (n == -1) then
    if (m <= 0 .or. mod(size(x), m) /= 0) error stop "ac_reshape 2d size mismatch"
    m_out = m
    n_out = size(x) / m_out
elseif (m == -1) then
    if (n <= 0 .or. mod(size(x), n) /= 0) error stop "ac_reshape 2d size mismatch"
    n_out = n
    m_out = size(x) / n_out
else
    n_out = n
    m_out = m
    if (n_out * m_out /= size(x)) error stop "ac_reshape 2d size mismatch"
end if

allocate(y(n_out, m_out))
k = 0
do i = 1, size(x, 1)
    do j = 1, size(x, 2)
        k = k + 1
        y((k - 1) / m_out + 1, mod(k - 1, m_out) + 1) = x(i, j)
    end do
end do
end function ac_reshape_2d_to_2d_int

pure function ac_add_axis_real(x) result(y)
real(dp), intent(in) :: x(:)
real(dp), allocatable :: y(:, :)
integer :: i

allocate(y(size(x), 1))
do i = 1, size(x)
    y(i, 1) = x(i)
end do
end function ac_add_axis_real

pure function ac_add_axis_int(x) result(y)
integer, intent(in) :: x(:)
integer, allocatable :: y(:, :)
integer :: i

allocate(y(size(x), 1))
do i = 1, size(x)
    y(i, 1) = x(i)
end do
end function ac_add_axis_int

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

pure function ac_mean_axis(x, axis) result(y)
real(dp), intent(in) :: x(:, :)
integer, intent(in) :: axis
real(dp), allocatable :: y(:)

if (axis == 0) then
    allocate(y(size(x, 2)))
    y = sum(x, dim=1) / real(size(x, 1), dp)
else
    allocate(y(size(x, 1)))
    y = sum(x, dim=2) / real(size(x, 2), dp)
end if
end function ac_mean_axis

pure function ac_min_axis_2d_real(x, axis) result(y)
real(dp), intent(in) :: x(:, :)
integer, intent(in) :: axis
real(dp), allocatable :: y(:)

if (axis == 0) then
    allocate(y(size(x, 2)))
    y = minval(x, dim=1)
else
    allocate(y(size(x, 1)))
    y = minval(x, dim=2)
end if
end function ac_min_axis_2d_real

pure function ac_min_axis_2d_int(x, axis) result(y)
integer, intent(in) :: x(:, :)
integer, intent(in) :: axis
integer, allocatable :: y(:)

if (axis == 0) then
    allocate(y(size(x, 2)))
    y = minval(x, dim=1)
else
    allocate(y(size(x, 1)))
    y = minval(x, dim=2)
end if
end function ac_min_axis_2d_int

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

pure function ac_argmin_1d_real(x) result(idx)
real(dp), intent(in) :: x(:)
integer :: idx
idx = minloc(x, dim=1) - 1
end function ac_argmin_1d_real

pure function ac_argmin_1d_int(x) result(idx)
integer, intent(in) :: x(:)
integer :: idx
idx = minloc(x, dim=1) - 1
end function ac_argmin_1d_int

pure function ac_argmin_2d_real(x) result(idx)
real(dp), intent(in) :: x(:, :)
integer :: idx
integer :: flat_idx(1)
flat_idx = minloc(reshape(x, [size(x)]))
idx = flat_idx(1) - 1
end function ac_argmin_2d_real

pure function ac_argmin_2d_int(x) result(idx)
integer, intent(in) :: x(:, :)
integer :: idx
integer :: flat_idx(1)
flat_idx = minloc(reshape(x, [size(x)]))
idx = flat_idx(1) - 1
end function ac_argmin_2d_int

pure function ac_argmax_1d_real(x) result(idx)
real(dp), intent(in) :: x(:)
integer :: idx
idx = maxloc(x, dim=1) - 1
end function ac_argmax_1d_real

pure function ac_argmax_1d_int(x) result(idx)
integer, intent(in) :: x(:)
integer :: idx
idx = maxloc(x, dim=1) - 1
end function ac_argmax_1d_int

pure function ac_argmax_2d_real(x) result(idx)
real(dp), intent(in) :: x(:, :)
integer :: idx
integer :: flat_idx(1)
flat_idx = maxloc(reshape(x, [size(x)]))
idx = flat_idx(1) - 1
end function ac_argmax_2d_real

pure function ac_argmax_2d_int(x) result(idx)
integer, intent(in) :: x(:, :)
integer :: idx
integer :: flat_idx(1)
flat_idx = maxloc(reshape(x, [size(x)]))
idx = flat_idx(1) - 1
end function ac_argmax_2d_int

pure function ac_argmin_axis_2d_real(x, axis) result(y)
real(dp), intent(in) :: x(:, :)
integer, intent(in) :: axis
integer, allocatable :: y(:)
integer :: i

if (axis == 0) then
    allocate(y(size(x, 2)))
    do i = 1, size(x, 2)
        y(i) = minloc(x(:, i), dim=1) - 1
    end do
else
    allocate(y(size(x, 1)))
    do i = 1, size(x, 1)
        y(i) = minloc(x(i, :), dim=1) - 1
    end do
end if
end function ac_argmin_axis_2d_real

pure function ac_argmin_axis_2d_int(x, axis) result(y)
integer, intent(in) :: x(:, :)
integer, intent(in) :: axis
integer, allocatable :: y(:)
integer :: i

if (axis == 0) then
    allocate(y(size(x, 2)))
    do i = 1, size(x, 2)
        y(i) = minloc(x(:, i), dim=1) - 1
    end do
else
    allocate(y(size(x, 1)))
    do i = 1, size(x, 1)
        y(i) = minloc(x(i, :), dim=1) - 1
    end do
end if
end function ac_argmin_axis_2d_int

pure function ac_argmax_axis_2d_real(x, axis) result(y)
real(dp), intent(in) :: x(:, :)
integer, intent(in) :: axis
integer, allocatable :: y(:)
integer :: i

if (axis == 0) then
    allocate(y(size(x, 2)))
    do i = 1, size(x, 2)
        y(i) = maxloc(x(:, i), dim=1) - 1
    end do
else
    allocate(y(size(x, 1)))
    do i = 1, size(x, 1)
        y(i) = maxloc(x(i, :), dim=1) - 1
    end do
end if
end function ac_argmax_axis_2d_real

pure function ac_argmax_axis_2d_int(x, axis) result(y)
integer, intent(in) :: x(:, :)
integer, intent(in) :: axis
integer, allocatable :: y(:)
integer :: i

if (axis == 0) then
    allocate(y(size(x, 2)))
    do i = 1, size(x, 2)
        y(i) = maxloc(x(:, i), dim=1) - 1
    end do
else
    allocate(y(size(x, 1)))
    do i = 1, size(x, 1)
        y(i) = maxloc(x(i, :), dim=1) - 1
    end do
end if
end function ac_argmax_axis_2d_int

pure function ac_var_1d_real(x, ddof) result(y)
real(dp), intent(in) :: x(:)
integer, intent(in) :: ddof
real(dp) :: y
real(dp) :: mean_x
integer :: n

n = size(x)
if (n == 0 .or. n - ddof <= 0) then
    y = 0.0_dp
    return
end if
mean_x = sum(x) / real(n, dp)
y = sum((x - mean_x)**2) / real(n - ddof, dp)
end function ac_var_1d_real

pure function ac_std_1d_real(x, ddof) result(y)
real(dp), intent(in) :: x(:)
integer, intent(in) :: ddof
real(dp) :: y

y = sqrt(ac_var_1d_real(x, ddof))
end function ac_std_1d_real

pure function ac_cumsum_1d_real(x) result(y)
real(dp), intent(in) :: x(:)
real(dp), allocatable :: y(:)
integer :: i

allocate(y(size(x)))
if (size(x) == 0) return
y(1) = x(1)
do i = 2, size(x)
    y(i) = y(i - 1) + x(i)
end do
end function ac_cumsum_1d_real

pure function ac_cumsum_1d_int(x) result(y)
integer, intent(in) :: x(:)
integer, allocatable :: y(:)
integer :: i

allocate(y(size(x)))
if (size(x) == 0) return
y(1) = x(1)
do i = 2, size(x)
    y(i) = y(i - 1) + x(i)
end do
end function ac_cumsum_1d_int

pure function ac_cumprod_1d_real(x) result(y)
real(dp), intent(in) :: x(:)
real(dp), allocatable :: y(:)
integer :: i

allocate(y(size(x)))
if (size(x) == 0) return
y(1) = x(1)
do i = 2, size(x)
    y(i) = y(i - 1) * x(i)
end do
end function ac_cumprod_1d_real

pure function ac_cumprod_1d_int(x) result(y)
integer, intent(in) :: x(:)
integer, allocatable :: y(:)
integer :: i

allocate(y(size(x)))
if (size(x) == 0) return
y(1) = x(1)
do i = 2, size(x)
    y(i) = y(i - 1) * x(i)
end do
end function ac_cumprod_1d_int

pure function ac_diff_1d_real(x) result(y)
real(dp), intent(in) :: x(:)
real(dp), allocatable :: y(:)

if (size(x) <= 1) then
    allocate(y(0))
    return
end if
allocate(y(size(x) - 1))
y = x(2:) - x(:size(x) - 1)
end function ac_diff_1d_real

pure function ac_gradient_1d_real(x) result(y)
real(dp), intent(in) :: x(:)
real(dp), allocatable :: y(:)
integer :: i

allocate(y(size(x)))
if (size(x) == 0) return
if (size(x) == 1) then
    y(1) = 0.0_dp
    return
end if
y(1) = x(2) - x(1)
do i = 2, size(x) - 1
    y(i) = 0.5_dp * (x(i + 1) - x(i - 1))
end do
y(size(x)) = x(size(x)) - x(size(x) - 1)
end function ac_gradient_1d_real

pure function ac_transpose_2d_real(x) result(y)
real(dp), intent(in) :: x(:, :)
real(dp), allocatable :: y(:, :)
y = transpose(x)
end function ac_transpose_2d_real

pure function ac_transpose_perm_3d_real(x, a1, a2, a3) result(y)
real(dp), intent(in) :: x(:, :, :)
integer, intent(in) :: a1, a2, a3
real(dp), allocatable :: y(:, :, :)
integer :: perm(3), dims(3), i, j, k

perm = [a1 + 1, a2 + 1, a3 + 1]
dims = [size(x, 1), size(x, 2), size(x, 3)]
allocate(y(dims(perm(1)), dims(perm(2)), dims(perm(3))))
do i = 1, size(x, 1)
    do j = 1, size(x, 2)
        do k = 1, size(x, 3)
            y(ac_index3(i, j, k, perm, 1), ac_index3(i, j, k, perm, 2), ac_index3(i, j, k, perm, 3)) = x(i, j, k)
        end do
    end do
end do
end function ac_transpose_perm_3d_real

pure function ac_transpose_perm_3d_int(x, a1, a2, a3) result(y)
integer, intent(in) :: x(:, :, :)
integer, intent(in) :: a1, a2, a3
integer, allocatable :: y(:, :, :)
integer :: perm(3), dims(3), i, j, k

perm = [a1 + 1, a2 + 1, a3 + 1]
dims = [size(x, 1), size(x, 2), size(x, 3)]
allocate(y(dims(perm(1)), dims(perm(2)), dims(perm(3))))
do i = 1, size(x, 1)
    do j = 1, size(x, 2)
        do k = 1, size(x, 3)
            y(ac_index3(i, j, k, perm, 1), ac_index3(i, j, k, perm, 2), ac_index3(i, j, k, perm, 3)) = x(i, j, k)
        end do
    end do
end do
end function ac_transpose_perm_3d_int

pure function ac_swapaxes_3d_real(x, axis1, axis2) result(y)
real(dp), intent(in) :: x(:, :, :)
integer, intent(in) :: axis1, axis2
real(dp), allocatable :: y(:, :, :)
integer :: perm(3), tmp

perm = [1, 2, 3]
tmp = perm(axis1 + 1)
perm(axis1 + 1) = perm(axis2 + 1)
perm(axis2 + 1) = tmp
y = ac_transpose_perm_3d_real(x, perm(1) - 1, perm(2) - 1, perm(3) - 1)
end function ac_swapaxes_3d_real

pure function ac_swapaxes_3d_int(x, axis1, axis2) result(y)
integer, intent(in) :: x(:, :, :)
integer, intent(in) :: axis1, axis2
integer, allocatable :: y(:, :, :)
integer :: perm(3), tmp

perm = [1, 2, 3]
tmp = perm(axis1 + 1)
perm(axis1 + 1) = perm(axis2 + 1)
perm(axis2 + 1) = tmp
y = ac_transpose_perm_3d_int(x, perm(1) - 1, perm(2) - 1, perm(3) - 1)
end function ac_swapaxes_3d_int

pure function ac_matmul_2d_2d_real(a, b) result(c)
real(dp), intent(in) :: a(:, :)
real(dp), intent(in) :: b(:, :)
real(dp), allocatable :: c(:, :)
c = matmul(a, b)
end function ac_matmul_2d_2d_real

pure function ac_matmul_2d_1d_real(a, b) result(c)
real(dp), intent(in) :: a(:, :)
real(dp), intent(in) :: b(:)
real(dp), allocatable :: c(:)
c = matmul(a, b)
end function ac_matmul_2d_1d_real

pure function ac_matmul_1d_2d_real(a, b) result(c)
real(dp), intent(in) :: a(:)
real(dp), intent(in) :: b(:, :)
real(dp), allocatable :: c(:)
c = matmul(a, b)
end function ac_matmul_1d_2d_real

pure function ac_matmul_1d_1d_real(a, b) result(c)
real(dp), intent(in) :: a(:)
real(dp), intent(in) :: b(:)
real(dp) :: c
c = dot_product(a, b)
end function ac_matmul_1d_1d_real

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
! Compute the sample covariance with observations stored by row.
! x: observation matrix with one sample per row.
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
if (n > 1) cov = matmul(transpose(xc), xc) / real(n - 1, dp)
if (n <= 1) cov = 0.0_dp
end function ac_cov_rowvar_false

function ac_slogdet_sign(a) result(sign_value)
! Return the sign of det(a) from an LU factorization.
! a: square matrix whose determinant sign is needed.
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
! Return log(abs(det(a))) from an LU factorization.
! a: square matrix whose log absolute determinant is needed.
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

function ac_det(a) result(det_value)
! Return det(a) from an LU factorization.
! a: square matrix whose determinant is needed.
real(dp), intent(in) :: a(:, :)
real(dp) :: det_value
real(dp), allocatable :: lu(:, :)
integer :: i, swap_count

call ac_lu_factor(a, lu, swap_count)
det_value = merge(1.0_dp, -1.0_dp, modulo(swap_count, 2) == 0)
do i = 1, size(lu, 1)
    det_value = det_value * lu(i, i)
end do
end function ac_det

pure function ac_inv(a) result(inv)
! Compute a matrix inverse with pivoted Gauss-Jordan elimination.
! a: square matrix to invert.
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
! Load a whitespace-delimited numeric text matrix.
! path: input file path.
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

logical function ac_file_exists(path)
character(len=*), intent(in) :: path

inquire(file=trim(path), exist=ac_file_exists)
end function ac_file_exists

function ac_loadtxt_1d(path) result(x)
! Load one numeric value per nonblank line into a 1D real array.
character(len=*), intent(in) :: path
real(dp), allocatable :: x(:)
character(len=4096) :: line
integer :: unit, ios, nrow, i

nrow = 0
open(newunit=unit, file=trim(path), status="old", action="read")
do
    read(unit, '(A)', iostat=ios) line
    if (ios /= 0) exit
    if (len_trim(line) == 0) cycle
    nrow = nrow + 1
end do
rewind(unit)
allocate(x(nrow))
i = 0
do
    read(unit, '(A)', iostat=ios) line
    if (ios /= 0) exit
    if (len_trim(line) == 0) cycle
    i = i + 1
    read(line, *) x(i)
end do
close(unit)
end function ac_loadtxt_1d

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

pure function ac_sort_1d_real(x) result(y)
real(dp), intent(in) :: x(:)
real(dp), allocatable :: y(:)
real(dp) :: tmp
integer :: i, j

y = x
do i = 1, size(y) - 1
    do j = i + 1, size(y)
        if (y(j) < y(i)) then
            tmp = y(i)
            y(i) = y(j)
            y(j) = tmp
        end if
    end do
end do
end function ac_sort_1d_real

pure function ac_sort_1d_int(x) result(y)
integer, intent(in) :: x(:)
integer, allocatable :: y(:)
integer :: tmp
integer :: i, j

y = x
do i = 1, size(y) - 1
    do j = i + 1, size(y)
        if (y(j) < y(i)) then
            tmp = y(i)
            y(i) = y(j)
            y(j) = tmp
        end if
    end do
end do
end function ac_sort_1d_int

pure function ac_unique_1d_int(x) result(y)
integer, intent(in) :: x(:)
integer, allocatable :: y(:)
integer, allocatable :: tmp(:)
integer :: i, n

if (size(x) == 0) then
    allocate(y(0))
    return
end if
tmp = ac_sort_1d_int(x)
n = 1
do i = 2, size(tmp)
    if (tmp(i) /= tmp(i - 1)) n = n + 1
end do
allocate(y(n))
y(1) = tmp(1)
n = 1
do i = 2, size(tmp)
    if (tmp(i) /= tmp(i - 1)) then
        n = n + 1
        y(n) = tmp(i)
    end if
end do
end function ac_unique_1d_int

pure function ac_unique_1d_real(x) result(y)
real(dp), intent(in) :: x(:)
real(dp), allocatable :: y(:)
real(dp), allocatable :: tmp(:)
integer :: i, n

if (size(x) == 0) then
    allocate(y(0))
    return
end if
tmp = ac_sort_1d_real(x)
n = 1
do i = 2, size(tmp)
    if (tmp(i) /= tmp(i - 1)) n = n + 1
end do
allocate(y(n))
y(1) = tmp(1)
n = 1
do i = 2, size(tmp)
    if (tmp(i) /= tmp(i - 1)) then
        n = n + 1
        y(n) = tmp(i)
    end if
end do
end function ac_unique_1d_real

pure function ac_bincount_1d_int(x) result(y)
integer, intent(in) :: x(:)
integer, allocatable :: y(:)
integer :: i

if (size(x) == 0) then
    allocate(y(0))
    return
end if
allocate(y(maxval(x) + 1))
y = 0
do i = 1, size(x)
    if (x(i) >= 0) y(x(i) + 1) = y(x(i) + 1) + 1
end do
end function ac_bincount_1d_int

pure function ac_searchsorted_left_1d_int(a, v) result(idx)
integer, intent(in) :: a(:), v(:)
integer, allocatable :: idx(:)
integer :: i, j

allocate(idx(size(v)))
do i = 1, size(v)
    idx(i) = size(a)
    do j = 1, size(a)
        if (a(j) >= v(i)) then
            idx(i) = j - 1
            exit
        end if
    end do
end do
end function ac_searchsorted_left_1d_int

pure function ac_searchsorted_right_1d_int(a, v) result(idx)
integer, intent(in) :: a(:), v(:)
integer, allocatable :: idx(:)
integer :: i, j

allocate(idx(size(v)))
do i = 1, size(v)
    idx(i) = size(a)
    do j = 1, size(a)
        if (a(j) > v(i)) then
            idx(i) = j - 1
            exit
        end if
    end do
end do
end function ac_searchsorted_right_1d_int

pure function ac_reduceat_add_1d_real(x, idx) result(y)
real(dp), intent(in) :: x(:)
integer, intent(in) :: idx(:)
real(dp), allocatable :: y(:)
integer :: i, lo, hi, n, m

n = size(x)
m = size(idx)
allocate(y(m))
do i = 1, m
    lo = idx(i) + 1
    if (lo < 1 .or. lo > n) error stop "ac_reduceat_add_1d_real: idx out of bounds"
    if (i < m) then
        hi = idx(i + 1)
    else
        hi = n
    end if
    hi = max(lo, min(hi, n))
    y(i) = sum(x(lo:hi))
end do
end function ac_reduceat_add_1d_real

pure function ac_reduceat_add_1d_int(x, idx) result(y)
integer, intent(in) :: x(:)
integer, intent(in) :: idx(:)
integer, allocatable :: y(:)
integer :: i, lo, hi, n, m

n = size(x)
m = size(idx)
allocate(y(m))
do i = 1, m
    lo = idx(i) + 1
    if (lo < 1 .or. lo > n) error stop "ac_reduceat_add_1d_int: idx out of bounds"
    if (i < m) then
        hi = idx(i + 1)
    else
        hi = n
    end if
    hi = max(lo, min(hi, n))
    y(i) = sum(x(lo:hi))
end do
end function ac_reduceat_add_1d_int

pure function ac_histogram_counts_1d_real(x, bins) result(h)
real(dp), intent(in) :: x(:), bins(:)
integer, allocatable :: h(:)
integer :: i, j, nb
logical :: placed

nb = size(bins) - 1
if (nb < 1) error stop "ac_histogram_counts_1d_real: bins must have at least 2 entries"
allocate(h(nb), source=0)
do i = 1, size(x)
    placed = .false.
    do j = 1, nb - 1
        if (x(i) >= bins(j) .and. x(i) < bins(j + 1)) then
            h(j) = h(j) + 1
            placed = .true.
            exit
        end if
    end do
    if (.not. placed) then
        if (x(i) >= bins(nb) .and. x(i) <= bins(nb + 1)) h(nb) = h(nb) + 1
    end if
end do
end function ac_histogram_counts_1d_real

pure function ac_histogram_edges_1d_real(bins) result(edges)
real(dp), intent(in) :: bins(:)
real(dp), allocatable :: edges(:)

edges = bins
end function ac_histogram_edges_1d_real

pure function ac_repeat_1d_int(x, reps) result(y)
integer, intent(in) :: x(:)
integer, intent(in) :: reps
integer, allocatable :: y(:)
integer :: i, j, k

allocate(y(size(x) * reps))
k = 0
do i = 1, size(x)
    do j = 1, reps
        k = k + 1
        y(k) = x(i)
end do
end do
end function ac_repeat_1d_int

pure function ac_repeat_scalar_real(x, reps) result(y)
real(dp), intent(in) :: x
integer, intent(in) :: reps
real(dp), allocatable :: y(:)

allocate(y(reps))
y = x
end function ac_repeat_scalar_real

pure function ac_repeat_scalar_int(x, reps) result(y)
integer, intent(in) :: x
integer, intent(in) :: reps
integer, allocatable :: y(:)

allocate(y(reps))
y = x
end function ac_repeat_scalar_int

pure function ac_repeat_1d_real(x, reps) result(y)
real(dp), intent(in) :: x(:)
integer, intent(in) :: reps
real(dp), allocatable :: y(:)
integer :: i, j, k

allocate(y(size(x) * reps))
k = 0
do i = 1, size(x)
    do j = 1, reps
        k = k + 1
        y(k) = x(i)
    end do
end do
end function ac_repeat_1d_real

pure function ac_repeat_axis_2d_int(x, reps, axis) result(y)
integer, intent(in) :: x(:, :)
integer, intent(in) :: reps, axis
integer, allocatable :: y(:, :)
integer :: i, j

if (axis == 0) then
    allocate(y(size(x, 1) * reps, size(x, 2)))
    do i = 1, size(x, 1)
        do j = 1, reps
            y((i - 1) * reps + j, :) = x(i, :)
        end do
    end do
else
    allocate(y(size(x, 1), size(x, 2) * reps))
    do i = 1, size(x, 2)
        do j = 1, reps
            y(:, (i - 1) * reps + j) = x(:, i)
        end do
    end do
end if
end function ac_repeat_axis_2d_int

pure function ac_repeat_axis_2d_real(x, reps, axis) result(y)
real(dp), intent(in) :: x(:, :)
integer, intent(in) :: reps, axis
real(dp), allocatable :: y(:, :)
integer :: i, j

if (axis == 0) then
    allocate(y(size(x, 1) * reps, size(x, 2)))
    do i = 1, size(x, 1)
        do j = 1, reps
            y((i - 1) * reps + j, :) = x(i, :)
        end do
    end do
else
    allocate(y(size(x, 1), size(x, 2) * reps))
    do i = 1, size(x, 2)
        do j = 1, reps
            y(:, (i - 1) * reps + j) = x(:, i)
        end do
    end do
end if
end function ac_repeat_axis_2d_real

pure function ac_tile_1d_int(x, reps) result(y)
integer, intent(in) :: x(:)
integer, intent(in) :: reps
integer, allocatable :: y(:)
integer :: i

allocate(y(size(x) * reps))
do i = 1, reps
    y((i - 1) * size(x) + 1:i * size(x)) = x
end do
end function ac_tile_1d_int

pure function ac_tile_1d_real(x, reps) result(y)
real(dp), intent(in) :: x(:)
integer, intent(in) :: reps
real(dp), allocatable :: y(:)
integer :: i

allocate(y(size(x) * reps))
do i = 1, reps
    y((i - 1) * size(x) + 1:i * size(x)) = x
end do
end function ac_tile_1d_real

pure function ac_tile_2d_int(x, reps) result(y)
integer, intent(in) :: x(:, :)
integer, intent(in) :: reps(:)
integer, allocatable :: y(:, :)
integer :: nr, nc, i, j

nr = reps(1)
nc = reps(2)
allocate(y(size(x, 1) * nr, size(x, 2) * nc))
do i = 1, nr
    do j = 1, nc
        y((i - 1) * size(x, 1) + 1:i * size(x, 1), (j - 1) * size(x, 2) + 1:j * size(x, 2)) = x
    end do
end do
end function ac_tile_2d_int

pure function ac_tile_2d_real(x, reps) result(y)
real(dp), intent(in) :: x(:, :)
integer, intent(in) :: reps(:)
real(dp), allocatable :: y(:, :)
integer :: nr, nc, i, j

nr = reps(1)
nc = reps(2)
allocate(y(size(x, 1) * nr, size(x, 2) * nc))
do i = 1, nr
    do j = 1, nc
        y((i - 1) * size(x, 1) + 1:i * size(x, 1), (j - 1) * size(x, 2) + 1:j * size(x, 2)) = x
    end do
end do
end function ac_tile_2d_real

pure function ac_diag_from_vec_int(v) result(x)
integer, intent(in) :: v(:)
integer, allocatable :: x(:, :)
integer :: i

allocate(x(size(v), size(v)))
x = 0
do i = 1, size(v)
    x(i, i) = v(i)
end do
end function ac_diag_from_vec_int

pure function ac_diag_from_size_int(n) result(x)
integer, intent(in) :: n
real(dp), allocatable :: x(:, :)
integer :: i

allocate(x(n, n))
x = 0.0_dp
do i = 1, n
    x(i, i) = 1.0_dp
end do
end function ac_diag_from_size_int

pure function ac_diag_from_size_real(n) result(x)
real(dp), intent(in) :: n
real(dp), allocatable :: x(:, :)
integer :: i, n_int

n_int = int(n)
allocate(x(n_int, n_int))
x = 0.0_dp
do i = 1, n_int
    x(i, i) = 1.0_dp
end do
end function ac_diag_from_size_real

pure function ac_diag_from_vec_real(v) result(x)
real(dp), intent(in) :: v(:)
real(dp), allocatable :: x(:, :)
integer :: i

allocate(x(size(v), size(v)))
x = 0.0_dp
do i = 1, size(v)
    x(i, i) = v(i)
end do
end function ac_diag_from_vec_real

pure function ac_diag_from_mat_int(a) result(v)
integer, intent(in) :: a(:, :)
integer, allocatable :: v(:)
integer :: n, i

n = min(size(a, 1), size(a, 2))
allocate(v(n))
do i = 1, n
    v(i) = a(i, i)
end do
end function ac_diag_from_mat_int

pure function ac_diag_from_mat_real(a) result(v)
real(dp), intent(in) :: a(:, :)
real(dp), allocatable :: v(:)
integer :: n, i

n = min(size(a, 1), size(a, 2))
allocate(v(n))
do i = 1, n
    v(i) = a(i, i)
end do
end function ac_diag_from_mat_real

pure function ac_diag_from_mat_real_k(a, k) result(v)
real(dp), intent(in) :: a(:, :)
integer, intent(in) :: k
real(dp), allocatable :: v(:)
integer :: i, n, row0, col0

if (k >= 0) then
    row0 = 1
    col0 = 1 + k
else
    row0 = 1 - k
    col0 = 1
end if
n = min(size(a, 1) - row0 + 1, size(a, 2) - col0 + 1)
if (n < 0) n = 0
allocate(v(n))
do i = 1, n
    v(i) = a(row0 + i - 1, col0 + i - 1)
end do
end function ac_diag_from_mat_real_k

pure function ac_triu_2d_real(a) result(out)
real(dp), intent(in) :: a(:, :)
real(dp), allocatable :: out(:, :)
integer :: i

out = a
do i = 2, size(a, 1)
    out(i, :min(i - 1, size(a, 2))) = 0.0_dp
end do
end function ac_triu_2d_real

pure function ac_tril_2d_real(a) result(out)
real(dp), intent(in) :: a(:, :)
real(dp), allocatable :: out(:, :)
integer :: i

out = a
do i = 1, min(size(a, 1), size(a, 2) - 1)
    out(i, i + 1:) = 0.0_dp
end do
end function ac_tril_2d_real

pure real(dp) function ac_trace_2d_real(a)
real(dp), intent(in) :: a(:, :)
integer :: i, n

n = min(size(a, 1), size(a, 2))
ac_trace_2d_real = 0.0_dp
do i = 1, n
    ac_trace_2d_real = ac_trace_2d_real + a(i, i)
end do
end function ac_trace_2d_real

pure function ac_outer_1d_real(x, y) result(out)
real(dp), intent(in) :: x(:), y(:)
real(dp), allocatable :: out(:, :)
integer :: i, j

allocate(out(size(x), size(y)))
do i = 1, size(x)
    do j = 1, size(y)
        out(i, j) = x(i) * y(j)
    end do
end do
end function ac_outer_1d_real

pure function ac_kron_1d_real(x, y) result(out)
real(dp), intent(in) :: x(:), y(:)
real(dp), allocatable :: out(:, :)
integer :: i, j

allocate(out(size(x), size(y)))
do i = 1, size(x)
    do j = 1, size(y)
        out(i, j) = x(i) * y(j)
    end do
end do
end function ac_kron_1d_real

pure function ac_concatenate_axis0_2d_real(a, b) result(out)
real(dp), intent(in) :: a(:, :), b(:, :)
real(dp), allocatable :: out(:, :)

allocate(out(size(a, 1) + size(b, 1), size(a, 2)))
out(1:size(a, 1), :) = a
out(size(a, 1) + 1:, :) = b
end function ac_concatenate_axis0_2d_real

pure function ac_concatenate_axis0_2d_int(a, b) result(out)
integer, intent(in) :: a(:, :), b(:, :)
integer, allocatable :: out(:, :)

allocate(out(size(a, 1) + size(b, 1), size(a, 2)))
out(1:size(a, 1), :) = a
out(size(a, 1) + 1:, :) = b
end function ac_concatenate_axis0_2d_int

pure function ac_concatenate_axis1_2d_real(a, b) result(out)
real(dp), intent(in) :: a(:, :), b(:, :)
real(dp), allocatable :: out(:, :)

allocate(out(size(a, 1), size(a, 2) + size(b, 2)))
out(:, 1:size(a, 2)) = a
out(:, size(a, 2) + 1:) = b
end function ac_concatenate_axis1_2d_real

pure function ac_concatenate_axis1_2d_int(a, b) result(out)
integer, intent(in) :: a(:, :), b(:, :)
integer, allocatable :: out(:, :)

allocate(out(size(a, 1), size(a, 2) + size(b, 2)))
out(:, 1:size(a, 2)) = a
out(:, size(a, 2) + 1:) = b
end function ac_concatenate_axis1_2d_int

pure function ac_hstack_2d_real(a, b) result(out)
real(dp), intent(in) :: a(:, :), b(:, :)
real(dp), allocatable :: out(:, :)

out = ac_concatenate_axis1_2d_real(a, b)
end function ac_hstack_2d_real

pure function ac_hstack_2d_int(a, b) result(out)
integer, intent(in) :: a(:, :), b(:, :)
integer, allocatable :: out(:, :)

out = ac_concatenate_axis1_2d_int(a, b)
end function ac_hstack_2d_int

pure function ac_vstack_2d_real(a, b) result(out)
real(dp), intent(in) :: a(:, :), b(:, :)
real(dp), allocatable :: out(:, :)

out = ac_concatenate_axis0_2d_real(a, b)
end function ac_vstack_2d_real

pure function ac_vstack_2d_int(a, b) result(out)
integer, intent(in) :: a(:, :), b(:, :)
integer, allocatable :: out(:, :)

out = ac_concatenate_axis0_2d_int(a, b)
end function ac_vstack_2d_int

pure function ac_stack_axis0_2d_real(a, b) result(out)
real(dp), intent(in) :: a(:, :), b(:, :)
real(dp), allocatable :: out(:, :, :)

allocate(out(2, size(a, 1), size(a, 2)))
out(1, :, :) = a
out(2, :, :) = b
end function ac_stack_axis0_2d_real

pure function ac_stack_axis0_2d_int(a, b) result(out)
integer, intent(in) :: a(:, :), b(:, :)
integer, allocatable :: out(:, :, :)

allocate(out(2, size(a, 1), size(a, 2)))
out(1, :, :) = a
out(2, :, :) = b
end function ac_stack_axis0_2d_int

pure function ac_stack_axis2_2d_real(a, b) result(out)
real(dp), intent(in) :: a(:, :), b(:, :)
real(dp), allocatable :: out(:, :, :)

allocate(out(size(a, 1), size(a, 2), 2))
out(:, :, 1) = a
out(:, :, 2) = b
end function ac_stack_axis2_2d_real

pure function ac_stack_axis2_2d_int(a, b) result(out)
integer, intent(in) :: a(:, :), b(:, :)
integer, allocatable :: out(:, :, :)

allocate(out(size(a, 1), size(a, 2), 2))
out(:, :, 1) = a
out(:, :, 2) = b
end function ac_stack_axis2_2d_int

function ac_argsort_real(x) result(idx)
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
end function ac_argsort_real

function ac_argsort_int(x) result(idx)
integer, intent(in) :: x(:)
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
end function ac_argsort_int

pure function ac_float_scalar(x) result(y)
real(dp), intent(in) :: x
real(dp) :: y
y = x
end function ac_float_scalar

pure function ac_float_int_scalar(x) result(y)
integer, intent(in) :: x
real(dp) :: y
y = real(x, dp)
end function ac_float_int_scalar

pure function ac_float_1x1(x) result(y)
real(dp), intent(in) :: x(:, :)
real(dp) :: y
y = x(1, 1)
end function ac_float_1x1

subroutine ac_savetxt_1d(path, x, fmt)
character(len=*), intent(in) :: path
real(dp), intent(in) :: x(:)
character(len=*), intent(in) :: fmt
integer :: unit, j
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
do j = 1, size(x)
    write(unit, "(" // trim(item_fmt) // ")") x(j)
end do
close(unit)
end subroutine ac_savetxt_1d

subroutine ac_savetxt_1d_default(path, x)
character(len=*), intent(in) :: path
real(dp), intent(in) :: x(:)
call ac_savetxt_1d(path, x, "es24.16e3")
end subroutine ac_savetxt_1d_default

subroutine ac_savetxt_2d(path, x, fmt)
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
end subroutine ac_savetxt_2d

subroutine ac_savetxt_2d_default(path, x)
character(len=*), intent(in) :: path
real(dp), intent(in) :: x(:, :)
call ac_savetxt_2d(path, x, "es24.16e3")
end subroutine ac_savetxt_2d_default

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
! Count whitespace-delimited fields in a text line.
! line: input text line.
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
! Compute a lower-triangular Cholesky factor.
! a: symmetric positive-definite matrix.
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
! Compute an LU factorization with partial pivoting.
! a: square matrix to factor.
! lu: packed LU factors.
! swap_count: optional row-swap count.
! piv: optional pivot order.
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
! Solve LU x = b in place using packed LU factors.
! lu: packed LU factors.
! b: right-hand side overwritten by the solution.
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

pure subroutine ac_swap_rows(a, i, j)
! Swap two rows of a matrix.
! a: matrix to modify.
! i: first row index.
! j: second row index.
real(dp), intent(inout) :: a(:, :)
integer, intent(in) :: i, j
real(dp) :: tmp(size(a, 2))

tmp = a(i, :)
a(i, :) = a(j, :)
a(j, :) = tmp
end subroutine ac_swap_rows

subroutine ac_swap_scalars(x, y)
! Swap two real scalars.
! x: first scalar.
! y: second scalar.
real(dp), intent(inout) :: x, y
real(dp) :: tmp

tmp = x
x = y
y = tmp
end subroutine ac_swap_scalars

subroutine ac_swap_ints(x, y)
! Swap two integers.
! x: first integer.
! y: second integer.
integer, intent(inout) :: x, y
integer :: tmp

tmp = x
x = y
y = tmp
end subroutine ac_swap_ints

pure integer function ac_index3(i, j, k, perm, axis)
integer, intent(in) :: i, j, k, perm(3), axis
integer :: vals(3)

vals = [i, j, k]
ac_index3 = vals(perm(axis))
end function ac_index3

function ac_wall_time() result(out)
real(dp) :: out
integer :: count, rate

call system_clock(count, rate)
if (rate > 0) then
    out = real(count, dp) / real(rate, dp)
else
    call cpu_time(out)
end if
end function ac_wall_time

end module ac_numpy_mod
