module ac_string_mod
use kind_mod, only: dp
implicit none
private
public :: ac_lower, ac_format_default, ac_format_fixed, ac_format_scientific, &
    & ac_format_int, ac_format_array, ac_set_printoptions

interface ac_format_default
    module procedure ac_format_default_real
    module procedure ac_format_default_int
    module procedure ac_format_default_logical
end interface ac_format_default

interface ac_format_array
    module procedure ac_format_array_1d_real
end interface ac_format_array

integer :: ac_print_precision = 6
logical :: ac_print_suppress = .true.
integer :: ac_print_linewidth = 80

contains

pure elemental function ac_lower(value) result(out)
! Convert ASCII uppercase letters in value to lowercase.
! value: input character string.
character(len=*), intent(in) :: value
character(len=len(value)) :: out
integer :: i, code

out = value
do i = 1, len(value)
    code = iachar(value(i:i))
    if (code >= iachar("A") .and. code <= iachar("Z")) out(i:i) = achar(code + 32)
end do
end function ac_lower

subroutine ac_set_printoptions(precision, suppress, linewidth, use_precision, use_suppress, use_linewidth)
integer, intent(in) :: precision, linewidth
logical, intent(in) :: suppress, use_precision, use_suppress, use_linewidth

if (use_precision) ac_print_precision = precision
if (use_suppress) ac_print_suppress = suppress
if (use_linewidth) ac_print_linewidth = linewidth
end subroutine ac_set_printoptions

function ac_format_default_real(value) result(out)
real(dp), intent(in) :: value
character(len=:), allocatable :: out
character(len=64) :: buffer
character(len=32) :: fmt

if (ac_print_precision >= 0) then
    if (ac_print_suppress) then
        write(fmt, '("(f0.", i0, ")")') ac_print_precision
    else
        write(fmt, '("(g0.", i0, ")")') ac_print_precision
    end if
    write(buffer, fmt) value
else
    write(buffer, "(g0)") value
end if
out = trim(adjustl(buffer))
out = ac_normalize_decimal_string(out)
end function ac_format_default_real

function ac_format_default_int(value) result(out)
integer, intent(in) :: value
character(len=:), allocatable :: out
character(len=64) :: buffer

write(buffer, "(i0)") value
out = trim(adjustl(buffer))
end function ac_format_default_int

function ac_format_default_logical(value) result(out)
logical, intent(in) :: value
character(len=:), allocatable :: out

if (value) then
    out = "True"
else
    out = "False"
end if
end function ac_format_default_logical

function ac_format_fixed(value, decimals, width) result(out)
real(dp), intent(in) :: value
integer, intent(in) :: decimals
integer, intent(in), optional :: width
character(len=:), allocatable :: out
character(len=64) :: buffer
character(len=32) :: fmt

if (present(width)) then
    write(fmt, '("(f", i0, ".", i0, ")")') width, decimals
else
    write(fmt, '("(f0.", i0, ")")') decimals
end if
write(buffer, fmt) value
if (present(width)) then
    out = buffer(:len_trim(buffer))
else
    out = trim(adjustl(buffer))
end if
out = ac_normalize_decimal_string(out)
end function ac_format_fixed

function ac_format_scientific(value, decimals, width) result(out)
real(dp), intent(in) :: value
integer, intent(in) :: decimals
integer, intent(in), optional :: width
character(len=:), allocatable :: out
character(len=64) :: buffer
character(len=32) :: fmt

if (present(width)) then
    write(fmt, '("(es", i0, ".", i0, "e3)")') width, decimals
else
    write(fmt, '("(es24.", i0, "e3)")') decimals
end if
write(buffer, fmt) value
if (present(width)) then
    out = ac_lower(buffer(:len_trim(buffer)))
else
    out = trim(adjustl(ac_lower(buffer)))
end if
end function ac_format_scientific

function ac_format_int(value, width) result(out)
integer, intent(in) :: value, width
character(len=:), allocatable :: out
character(len=64) :: buffer
character(len=32) :: fmt

if (width <= 0) then
    fmt = "(i0)"
else
    write(fmt, '("(i", i0, ")")') width
end if
write(buffer, fmt) value
out = trim(buffer)
end function ac_format_int

function ac_format_array_1d_real(values) result(out)
real(dp), intent(in) :: values(:)
character(len=:), allocatable :: out
character(len=:), allocatable :: piece
integer :: i

out = "["
do i = 1, size(values)
    piece = ac_format_default_real(values(i))
    if (i > 1) then
        out = out // " "
        if (values(i) >= 0.0_dp) out = out // " "
    else if (values(i) >= 0.0_dp) then
        out = out // " "
    end if
    out = out // piece
end do
out = out // "]"
end function ac_format_array_1d_real

pure function ac_normalize_decimal_string(value) result(out)
character(len=*), intent(in) :: value
character(len=:), allocatable :: out

out = value
if (len(out) >= 1 .and. out(1:1) == ".") out = "0" // out
if (len(out) >= 2 .and. out(1:2) == "-.") out = "-0" // out(2:)
end function ac_normalize_decimal_string

end module ac_string_mod
