% quicksort_inplace.m
% In-place quicksort on an array in MATLAB.
% - If called with a filename (char/string), reads one number per line.
% - If called with a numeric vector, sorts that vector.
% - Prints the sorted numbers, one per line, to stdout.
%
% Usage (file):
%   matlab -batch "quicksort_inplace('input.txt')"
%   % where input.txt contains (e.g.)
%   % 5
%   % 3
%   % 8
%   % 1
%   % 2
%
% Usage (vector):
%   matlab -batch "quicksort_inplace([5 3 8 1 2])"
%
% Notes:
% - Implements Hoare partition scheme (stable enough for duplicates).
% - MATLAB recursion limit is ~500 by default; for very large inputs,
%   increase it (e.g., set(0,'RecursionLimit',10000)).

function quicksort_inplace(inputArg)
    if nargin < 1
        error('Provide a filename or a numeric vector.');
    end

    % -------- read input into A --------
    if isnumeric(inputArg)
        A = inputArg(:);  % column vector
    elseif ischar(inputArg) || isstring(inputArg)
        fname = char(inputArg);
        fid = fopen(fname, 'r');
        if fid < 0
            error('Cannot open file: %s', fname);
        end
        C = textscan(fid, '%f', 'Whitespace', " \b\t\r\n", 'CommentStyle', '#');
        fclose(fid);
        A = C{1};
    else
        error('Unsupported input type. Use filename or numeric vector.');
    end

    n = numel(A);
    if n == 0
        return;
    end

    % Optionally raise recursion limit for large arrays
    if n > 5000
        try
            set(0, 'RecursionLimit', max(get(0,'RecursionLimit'), 10000));
        catch
            % ignore if not permitted
        end
    end

    % -------- quicksort in-place on A --------
    qsort(1, n);

    % -------- output --------
    for i = 1:n
        fprintf('%g\n', A(i));
    end

    % ===== nested helpers (capture A by reference) =====

    function swap(i,j)
        tmp = A(i);
        A(i) = A(j);
        A(j) = tmp;
    end

    % Hoare partition; returns index j s.t. A(lo..j) <= pivot <= A(j+1..hi)
    function j = partition_hoare(lo, hi)
        mid = floor((lo + hi)/2);
        pivot = A(mid);
        i = lo - 1;
        j = hi + 1;
        while true
            % move i right
            i = i + 1;
            while A(i) < pivot
                i = i + 1;
            end
            % move j left
            j = j - 1;
            while A(j) > pivot
                j = j - 1;
            end
            if i >= j
                return;
            end
            swap(i, j);
        end
    end

    % Tail-recursion–friendly quicksort (smaller side first)
    function qsort(lo, hi)
        while lo < hi
            p = partition_hoare(lo, hi);
            left_size  = p - lo + 1;
            right_size = hi - (p + 1) + 1;
            if left_size < right_size
                qsort(lo, p);
                lo = p + 1;   % tail call on right side
            else
                qsort(p + 1, hi);
                hi = p;       % tail call on left side
            end
        end
    end
end

