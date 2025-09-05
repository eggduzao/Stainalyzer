/* quicksort_inplace.c
 * In-place quicksort on an array (C99).
 *
 * - Reads numbers (ints or floats) one-per-line from a file path (argv[1]) or stdin.
 * - Ignores blank lines and lines starting with '#'.
 * - Sorts in-place using iterative quicksort with Hoare partition +
 *   median-of-three pivot selection; small ranges use insertion sort.
 * - Prints sorted values one per line (integers printed without ".0").
 *
 * Build:
 *   cc -O3 -std=c99 -Wall -Wextra -o quicksort_inplace quicksort_inplace.c
 *
 * Run:
 *   ./quicksort_inplace numbers.txt
 *   cat numbers.txt | ./quicksort_inplace
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <math.h>

#ifndef M_EPS
#define M_EPS (1e-9)
#endif

/* -------------------- Dynamic array of doubles -------------------- */
typedef struct {
    double *data;
    size_t  size;
    size_t  cap;
} Vec;

static void vec_init(Vec *v) {
    v->data = NULL; v->size = 0; v->cap = 0;
}

static void vec_reserve(Vec *v, size_t need) {
    if (need <= v->cap) return;
    size_t ncap = v->cap ? v->cap : 1024;
    while (ncap < need) ncap <<= 1;
    double *nd = (double*)realloc(v->data, ncap * sizeof(double));
    if (!nd) { perror("realloc"); exit(EXIT_FAILURE); }
    v->data = nd; v->cap = ncap;
}

static void vec_push(Vec *v, double x) {
    vec_reserve(v, v->size + 1);
    v->data[v->size++] = x;
}

static void vec_free(Vec *v) {
    free(v->data);
    v->data = NULL; v->size = v->cap = 0;
}

/* -------------------- Input helpers -------------------- */
static char *trim(char *s) {
    if (!s) return s;
    // ltrim
    while (*s && isspace((unsigned char)*s)) s++;
    // rtrim
    char *end = s + strlen(s);
    while (end > s && isspace((unsigned char)end[-1])) --end;
    *end = '\0';
    return s;
}

static int parse_double_strict(const char *s, double *out) {
    // Require full consumption of the token
    char *endp = NULL;
    double v = strtod(s, &endp);
    if (endp == s) return 0; // no parse
    // Skip trailing spaces
    while (*endp && isspace((unsigned char)*endp)) endp++;
    if (*endp != '\0') return 0; // junk at end
    if (!isfinite(v)) return 0;
    *out = v;
    return 1;
}

static void read_numbers(FILE *fp, Vec *v) {
    vec_init(v);
    char buf[4096];
    size_t lineno = 0;
    while (fgets(buf, sizeof buf, fp)) {
        lineno++;
        char *line = trim(buf);
        if (*line == '\0' || *line == '#') continue;
        double x;
        if (!parse_double_strict(line, &x)) {
            fprintf(stderr, "Invalid numeric line at %zu: %s\n", lineno, line);
            exit(EXIT_FAILURE);
        }
        vec_push(v, x);
    }
}

/* -------------------- Sorting primitives (0-based indexing) -------------------- */

static size_t median_of_three(double *a, size_t i, size_t j, size_t k) {
    double ai = a[i], aj = a[j], ak = a[k];
    if (ai < aj) {
        if (aj < ak) return j;
        else if (ai < ak) return i;
        else return k;
    } else {
        if (ai < ak) return i;
        else if (aj < ak) return j;
        else return k;
    }
}

static void insertion_sort(double *a, size_t lo, size_t hi) {
    if (hi <= lo) return;
    for (size_t i = lo + 1; i <= hi; ++i) {
        double key = a[i];
        size_t j = i;
        while (j > lo && a[j - 1] > key) {
            a[j] = a[j - 1];
            --j;
        }
        a[j] = key;
    }
}

/* Hoare partition. Returns index p such that [lo..p] <= pivot and [p+1..hi] >= pivot */
static size_t hoare_partition(double *a, size_t lo, size_t hi) {
    size_t mid = lo + (hi - lo) / 2;
    size_t pidx = median_of_three(a, lo, mid, hi);
    double pivot = a[pidx];

    size_t i = lo - 1;  // careful: will wrap for lo=0; we always ++ before read
    size_t j = hi + 1;
    for (;;) {
        do { ++i; } while (a[i] < pivot);
        do { --j; } while (a[j] > pivot);
        if (i >= j) return j;
        double tmp = a[i]; a[i] = a[j]; a[j] = tmp;
    }
}

typedef struct { size_t lo, hi; } Range;

static void quicksort_inplace(double *a, size_t n) {
    if (n < 2) return;

    const size_t SMALL = 24; // insertion sort cutoff
    // Manual stack
    size_t cap = 64, top = 0;
    Range *stack = (Range*)malloc(cap * sizeof(Range));
    if (!stack) { perror("malloc"); exit(EXIT_FAILURE); }
    #define PUSH(L,H) do{ if(top==cap){cap<<=1; stack=(Range*)realloc(stack,cap*sizeof(Range)); if(!stack){perror("realloc"); exit(EXIT_FAILURE);} } stack[top++] = (Range){(L),(H)}; }while(0)
    #define POP(R)    do{ (R)=stack[--top]; }while(0)

    PUSH(0, n - 1);

    while (top) {
        Range r; POP(r);
        size_t lo = r.lo, hi = r.hi;
        if (hi <= lo) continue;
        if (hi - lo + 1 <= SMALL) {
            insertion_sort(a, lo, hi);
            continue;
        }
        size_t p = hoare_partition(a, lo, hi);
        size_t left_sz  = p >= lo ? (p - lo + 1) : 0;
        size_t right_lo = p + 1, right_sz = hi >= right_lo ? (hi - right_lo + 1) : 0;

        // Tail recursion elimination strategy: push larger side first, pop smaller next
        if (left_sz > right_sz) {
            if (lo < p)      PUSH(lo, p);
            if (right_lo < hi) PUSH(right_lo, hi);
        } else {
            if (right_lo < hi) PUSH(right_lo, hi);
            if (lo < p)      PUSH(lo, p);
        }
    }
    free(stack);
    #undef PUSH
    #undef POP
}

/* -------------------- Output helpers -------------------- */
static void print_numbers(const double *a, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        double x = a[i];
        double rx = round(x);
        if (fabs(x - rx) < M_EPS) {
            // Looks like an integer; print without decimal
            printf("%.0f\n", x);
        } else {
            // Print in a non-scientific, trimmed fashion
            // Using %.15g is a decent compromise for doubles
            printf("%.15g\n", x);
        }
    }
}

/* -------------------- Main -------------------- */
int main(int argc, char **argv) {
    Vec v;
    if (argc >= 2) {
        FILE *fp = fopen(argv[1], "r");
        if (!fp) { perror("fopen"); return EXIT_FAILURE; }
        read_numbers(fp, &v);
        fclose(fp);
    } else {
        read_numbers(stdin, &v);
    }

    quicksort_inplace(v.data, v.size);
    print_numbers(v.data, v.size);
    vec_free(&v);
    return 0;
}

