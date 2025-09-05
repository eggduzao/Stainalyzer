! binsearch.f90 — Binary search (lower-bound) over a sorted ascending list.
! Build:
!   gfortran -O2 -std=f2008 -Wall -Wextra -o binsearch binsearch.f90
!
! Usage:
!   ./binsearch <path_or_-> <target_int>
!
!   <path_or_-> : path to a text file containing one integer per line.
!                 Use "-" to read from STDIN.
!   <target_int>: integer to search.
!
! Behavior:
!   • Skips blank lines and lines that can’t be parsed as integers
!     (emits one warning summary with a count).
!   • Verifies the list is in ascending (non-decreasing) order.
!   • Performs lower-bound binary search on 1-based indices.
!     - If found:   prints "FOUND <target> at index <1-based>"
!     - If absent:  prints "NOT FOUND <target>. Insertion index <1-based>, between <L> and <R>"
!                   (L/R become -inf/+inf at the extremes)

module bs_utils
  implicit none
  private
  public :: i64, push, lower_bound, check_sorted

  integer, parameter :: i64 = selected_int_kind(18)

contains

  subroutine push(val, arr, n, cap)
    integer(i64), intent(in)            :: val
    integer(i64), allocatable, intent(inout) :: arr(:)
    integer,        intent(inout)       :: n, cap
    integer(i64), allocatable :: tmp(:)
    if (n == cap) then
      if (cap == 0) then
        cap = 1024
        allocate(arr(cap))
      else
        cap = cap * 2
        allocate(tmp(cap))
        tmp(1:n) = arr(1:n)
        call move_alloc(tmp, arr)
      end if
    end if
    n = n + 1
    arr(n) = val
  end subroutine push

  pure function lower_bound(arr, n, tgt) result(pos)
    ! Return the first index in [1..n+1] where arr(pos) >= tgt.
    integer(i64), intent(in) :: arr(:)
    integer,      intent(in) :: n
    integer(i64), intent(in) :: tgt
    integer                  :: pos
    integer :: lo, hi, mid
    lo = 1
    hi = n + 1
    do while (lo < hi)
      mid = (lo + hi) / 2
      if (arr(mid) < tgt) then
        lo = mid + 1
      else
        hi = mid
      end if
    end do
    pos = lo
  end function lower_bound

  pure function check_sorted(arr, n, break_a, break_b) result(ok)
    integer(i64), intent(in)  :: arr(:)
    integer,      intent(in)  :: n
    integer(i64), intent(out) :: break_a, break_b
    logical :: ok
    integer :: i
    ok = .true.
    break_a = 0_i64
    break_b = 0_i64
    do i = 2, n
      if (arr(i) < arr(i-1)) then
        ok = .false.
        break_a = arr(i-1)
        break_b = arr(i)
        return
      end if
    end do
  end function check_sorted

end module bs_utils

program binsearch
  use bs_utils
  implicit none

  character(len=:), allocatable :: path
  character(len=4096) :: arg, line
  integer :: argc, ios, u, n, cap, nonint_count
  integer(i64), allocatable :: a(:)
  integer(i64) :: v, tgt
  logical :: ok
  integer(i64) :: ba, bb
  integer :: pos
  logical :: use_stdin

  call get_command_argument_count(argc)
  if (argc < 2) then
    write(*,*) 'Usage: ./binsearch <path_or_-> <target_int>'
    write(*,*) 'Example:'
    write(*,*) '  printf "1\n3\n4\n7\n9\n11\n15\n" | ./binsearch - 11'
    stop 1
  end if

  call get_command_argument(1, arg)
  path = trim(arg)
  call get_command_argument(2, arg)
  read(arg, *, iostat=ios) tgt
  if (ios /= 0) then
    write(*,*) 'ERROR: TARGET must be an integer, got: ', trim(arg)
    stop 1
  end if

  n = 0; cap = 0; nonint_count = 0
  allocate(a(0))  ! start empty

  use_stdin = (path == '-')

  if (.not. use_stdin) then
    open(newunit=u, file=path, status='old', action='read', iostat=ios)
    if (ios /= 0) then
      write(*,*) 'ERROR: Cannot open file: ', trim(path)
      stop 1
    end if
  else
    u = 5  ! stdin
  end if

  do
    read(u, '(A)', iostat=ios) line
    if (ios /= 0) exit
    if (len_trim(line) == 0) cycle
    ! Try to parse as integer
    read(line, *, iostat=ios) v
    if (ios == 0) then
      call push(v, a, n, cap)
    else
      nonint_count = nonint_count + 1
    end if
  end do

  if (.not. use_stdin) close(u)

  if (nonint_count > 0) then
    write(*,*) 'WARN: skipped ', nonint_count, ' non-integer line(s).'
  end if

  if (n == 0) then
    write(*,*) 'ERROR: No numeric input found.'
    stop 1
  end if

  ok = check_sorted(a, n, ba, bb)
  if (.not. ok) then
    write(*,'(A,1X,I0,1X,A,1X,I0)') 'ERROR: Input not in ascending order near', ba, 'then', bb
    stop 1
  end if

  pos = lower_bound(a, n, tgt)

  if (pos <= n .and. a(pos) == tgt) then
    write(*,'(A,1X,I0,1X,A,1X,I0)') 'FOUND', tgt, 'at index', pos
  else
    call print_insertion(a, n, pos, tgt)
  end if

contains

  subroutine print_insertion(arr, n, pos, tgt)
    integer(i64), intent(in) :: arr(:), tgt
    integer,      intent(in) :: n, pos
    character(len=*), parameter :: fmt = '(A,1X,I0,A,1X,I0,A,1X,A,1X,A)'
    character(len=32) :: lefts, rights
    if (pos > 1) then
      write(lefts, '(I0)') arr(pos-1)
    else
      lefts = '-inf'
    end if
    if (pos <= n) then
      write(rights, '(I0)') arr(pos)
    else
      rights = '+inf'
    end if
    write(*,fmt) 'NOT FOUND', tgt, '. Insertion index', pos, '(1-based), between', trim(lefts), 'and', trim(rights)
  end subroutine print_insertion

end program binsearch
