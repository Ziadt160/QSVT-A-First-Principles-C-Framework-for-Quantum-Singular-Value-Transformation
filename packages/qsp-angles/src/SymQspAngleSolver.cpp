// VENDORED from the QSVT_Project root (src/SymQspAngleSolver.cpp).
// This file has no third-party-library dependency (the symmetric-QSP Newton
// core is now a self-contained, standard-library-only implementation); it is
// byte-for-byte identical to the canonical source apart from this vendored
// comment block. Keep in sync with the root.

#include "SymQspAngleSolver.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <functional>
#include <vector>

namespace qsvt {

namespace {

using std::array;
using cd = std::complex<double>;

constexpr double kPi = 3.14159265358979323846;

// Newton convergence criterion and iteration cap, matching the validated dev
// harness.
constexpr double kNewtonCrit = 1e-12;
constexpr int kNewtonMaxIter = 100;

// ---------------------------------------------------------------------------
// Tiny 3x3 / 3-vector linear algebra, used by Rz2 and jacComponents. Row-major
// std::array<double,9> matrices; std::array<double,3> vectors. Only the handful
// of products the algorithm needs are implemented; each does the same scalar
// arithmetic a general dense-matrix library would, so results are identical.
// ---------------------------------------------------------------------------
using Mat3 = array<double, 9>;
using Vec3 = array<double, 3>;

inline double m3(const Mat3& a, int r, int c) { return a[3 * r + c]; }

// 3x3 * 3x3
inline Mat3 mul(const Mat3& a, const Mat3& b)
{
    Mat3 out{};
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
            double s = 0.0;
            for (int k = 0; k < 3; ++k) s += a[3 * i + k] * b[3 * k + j];
            out[3 * i + j] = s;
        }
    return out;
}

// 3x3 * 3-col-vector
inline Vec3 mul(const Mat3& a, const Vec3& v)
{
    Vec3 out{};
    for (int i = 0; i < 3; ++i)
        out[i] = a[3 * i + 0] * v[0] + a[3 * i + 1] * v[1] + a[3 * i + 2] * v[2];
    return out;
}

// 3-row-vector * 3x3 -> 3-row-vector (returned as Vec3)
inline Vec3 mul(const Vec3& rowv, const Mat3& a)
{
    Vec3 out{};
    for (int j = 0; j < 3; ++j)
        out[j] = rowv[0] * a[0 + j] + rowv[1] * a[3 + j] + rowv[2] * a[6 + j];
    return out;
}

// row . col (dot product)
inline double dot(const Vec3& a, const Vec3& b)
{
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

Mat3 Rz2(double phi)
{
    const double c = std::cos(2 * phi), s = std::sin(2 * phi);
    // [[c,-s,0],[s,c,0],[0,0,1]]
    return Mat3{c, -s, 0.0, s, c, 0.0, 0.0, 0.0, 1.0};
}

// ---------------------------------------------------------------------------
// Self-contained complex FFT: iterative radix-2 (for power-of-two lengths) plus
// a Bluestein (chirp-z) wrapper for the arbitrary length 4n that the Jacobian
// build requires. Forward transform uses the exp(-2*pi*i*k*j/N) convention and
// is UNNORMALIZED (the caller divides by N afterward), so it matches a plain
// forward DFT exactly. The Bluestein path can compute any N exactly -- we never
// zero-pad the length-N transform itself.
// ---------------------------------------------------------------------------

// In-place iterative radix-2 Cooley-Tukey. n must be a power of two. inverse =
// false -> forward (exp(-i...)), unnormalized.
void fftRadix2(std::vector<cd>& a, bool inverse)
{
    const int n = static_cast<int>(a.size());
    if (n <= 1) return;
    // Bit-reversal permutation.
    for (int i = 1, j = 0; i < n; ++i) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) std::swap(a[i], a[j]);
    }
    const double sign = inverse ? 1.0 : -1.0;
    for (int len = 2; len <= n; len <<= 1) {
        const double ang = sign * 2.0 * kPi / len;
        const cd wlen(std::cos(ang), std::sin(ang));
        for (int i = 0; i < n; i += len) {
            cd w(1.0, 0.0);
            const int half = len >> 1;
            for (int k = 0; k < half; ++k) {
                const cd u = a[i + k];
                const cd v = a[i + k + half] * w;
                a[i + k] = u + v;
                a[i + k + half] = u - v;
                w *= wlen;
            }
        }
    }
}

// Forward DFT of arbitrary length N via Bluestein's algorithm. Returns the full
// length-N complex spectrum out[k] = sum_j in[j] exp(-2*pi*i*k*j/N), i.e. an
// unnormalized forward DFT. The internal convolution is padded to a power of two
// >= 2N-1, but the transform LENGTH is exactly N -- the coefficient indexing the
// caller relies on is preserved.
std::vector<cd> dftBluestein(const std::vector<cd>& in)
{
    const int N = static_cast<int>(in.size());
    if (N == 0) return {};
    if (N == 1) return {in[0]};

    // Chirp w[j] = exp(-i * pi * j^2 / N). Use j^2 mod 2N to keep the angle
    // small and accurate for large N (j^2 can overflow double's exact-integer
    // range well before N is large, but j*j mod 2N stays in [0, 2N) exactly via
    // 64-bit integer arithmetic).
    std::vector<cd> w(N);
    for (int j = 0; j < N; ++j) {
        const long long jj = (static_cast<long long>(j) * j) % (2LL * N);
        const double ang = -kPi * static_cast<double>(jj) / N;
        w[j] = cd(std::cos(ang), std::sin(ang));
    }

    // Smallest power of two >= 2N - 1.
    int M = 1;
    while (M < 2 * N - 1) M <<= 1;

    std::vector<cd> A(M, cd(0.0, 0.0)), B(M, cd(0.0, 0.0));
    for (int j = 0; j < N; ++j) A[j] = in[j] * w[j];
    // B holds conj(w) as a symmetric two-sided sequence so the linear
    // convolution A*B reproduces sum_j in[j] w[j] conj(w[j-k]).
    B[0] = std::conj(w[0]);
    for (int j = 1; j < N; ++j) {
        const cd c = std::conj(w[j]);
        B[j] = c;
        B[M - j] = c;
    }

    fftRadix2(A, false);
    fftRadix2(B, false);
    for (int i = 0; i < M; ++i) A[i] *= B[i];
    fftRadix2(A, true);            // inverse, still unnormalized
    const double invM = 1.0 / M;   // normalize the inverse here

    std::vector<cd> out(N);
    for (int k = 0; k < N; ++k) out[k] = (A[k] * invM) * w[k];
    return out;
}

// Forward DFT of a real signal, full length-N complex spectrum (the real input
// is embedded as complex with zero imaginary part). Dispatches to radix-2 when N
// is a power of two, else Bluestein.
std::vector<cd> fftFwdReal(const std::vector<double>& re)
{
    const int N = static_cast<int>(re.size());
    std::vector<cd> a(N);
    for (int i = 0; i < N; ++i) a[i] = cd(re[i], 0.0);
    if (N > 1 && (N & (N - 1)) == 0) {
        fftRadix2(a, false);
        return a;
    }
    return dftBluestein(a);
}

// ---------------------------------------------------------------------------
// Dense n x n linear solve A x = b via Gaussian elimination with partial
// pivoting -- the same factorization a standard partial-pivot LU solver uses, so
// it yields the same x. A is row-major flattened (size n*n); b has size n.
// ---------------------------------------------------------------------------
std::vector<double> solveLinear(std::vector<double> A, std::vector<double> b, int n)
{
    for (int col = 0; col < n; ++col) {
        // Partial pivot: largest |A[row][col]| at or below the diagonal.
        int piv = col;
        double best = std::abs(A[col * n + col]);
        for (int r = col + 1; r < n; ++r) {
            const double v = std::abs(A[r * n + col]);
            if (v > best) {
                best = v;
                piv = r;
            }
        }
        if (piv != col) {
            for (int c = 0; c < n; ++c) std::swap(A[col * n + c], A[piv * n + c]);
            std::swap(b[col], b[piv]);
        }
        const double diag = A[col * n + col];
        // Eliminate below.
        for (int r = col + 1; r < n; ++r) {
            const double f = A[r * n + col] / diag;
            if (f == 0.0) continue;
            for (int c = col; c < n; ++c) A[r * n + c] -= f * A[col * n + c];
            b[r] -= f * b[col];
        }
    }
    // Back-substitution.
    std::vector<double> x(n);
    for (int r = n - 1; r >= 0; --r) {
        double s = b[r];
        for (int c = r + 1; c < n; ++c) s -= A[r * n + c] * x[c];
        x[r] = s / A[r * n + r];
    }
    return x;
}

// ---------------------------------------------------------------------------
// 2x2 complex matrix helpers backing the self-contained response
// Re<0|U(x, phases)|0> (defined as SymQspAngleSolver::response below). This
// replicates QspAngleSolver::response with a plain 2x2 complex matrix product,
// keeping the sym_qsp core independent of QspAngleSolver. 2x2 complex matrices
// are std::array<cd,4> in row-major order: [m00, m01, m10, m11].
// ---------------------------------------------------------------------------
using Mat2c = array<cd, 4>;

inline Mat2c mul2(const Mat2c& a, const Mat2c& b)
{
    return Mat2c{a[0] * b[0] + a[1] * b[2], a[0] * b[1] + a[1] * b[3],
                 a[2] * b[0] + a[3] * b[2], a[2] * b[1] + a[3] * b[3]};
}

// y of length n+1: first n are d(response)/d(reduced_k), last is the response.
std::vector<double> jacComponents(double a, const std::vector<double>& red, int parity)
{
    const int n = static_cast<int>(red.size());
    const double t = std::acos(std::max(-1.0, std::min(1.0, a)));
    const double c2 = std::cos(2 * t), s2 = std::sin(2 * t);
    // B = [[c2,0,-s2],[0,1,0],[s2,0,c2]]
    const Mat3 B{c2, 0.0, -s2, 0.0, 1.0, 0.0, s2, 0.0, c2};

    // L[k] are row vectors; L[n-1] = (0,1,0); L[k] = L[k+1] * Rz2(red[k+1]) * B.
    std::vector<Vec3> L(n);
    L[n - 1] = Vec3{0.0, 1.0, 0.0};
    for (int k = n - 2; k >= 0; --k)
        L[k] = mul(mul(L[k + 1], Rz2(red[k + 1])), B);

    // R[k] are column vectors; R[0] depends on parity; R[k] = B * (Rz2(red[k-1]) * R[k-1]).
    std::vector<Vec3> R(n);
    if (parity == 0)
        R[0] = Vec3{1.0, 0.0, 0.0};
    else
        R[0] = Vec3{std::cos(t), 0.0, std::sin(t)};
    for (int k = 1; k < n; ++k)
        R[k] = mul(B, mul(Rz2(red[k - 1]), R[k - 1]));

    std::vector<double> y(n + 1);
    for (int k = 0; k < n; ++k) {
        const double ph = 2 * red[k];
        // dRz = [[-sin,-cos,0],[cos,-sin,0],[0,0,0]]
        const Mat3 dRz{-std::sin(ph), -std::cos(ph), 0.0,
                       std::cos(ph), -std::sin(ph), 0.0,
                       0.0, 0.0, 0.0};
        // (L[k] * dRz * R[k]) scalar, times 2.
        y[k] = 2.0 * dot(mul(L[k], dRz), R[k]);
    }
    // (L[n-1] * Rz2(red[n-1]) * R[n-1]) scalar.
    y[n] = dot(mul(L[n - 1], Rz2(red[n - 1])), R[n - 1]);
    return y;
}

// Returns (F, dF): achieved reduced Chebyshev coeffs and Jacobian wrt reduced.
// dF is row-major flattened, size n*n.
void genJacobian(const std::vector<double>& red, int parity,
                 std::vector<double>& F, std::vector<double>& dF)
{
    const int n = static_cast<int>(red.size());
    const int dd = 2 * n;
    const int rows = 2 * dd; // = 4n
    const int cols = n + 1;
    // M is (4n) x (n+1), row-major.
    std::vector<double> M(static_cast<std::size_t>(rows) * cols, 0.0);
    auto Mref = [&](int r, int c) -> double& { return M[static_cast<std::size_t>(r) * cols + c]; };

    for (int r = 0; r <= n; ++r) {
        const double theta = r * kPi / dd;
        const std::vector<double> y = jacComponents(std::cos(theta), red, parity);
        for (int c = 0; c < cols; ++c) Mref(r, c) = y[c];
    }
    const double sgn = (parity % 2 == 0) ? 1.0 : -1.0;
    for (int i = 0; i < n; ++i)
        for (int c = 0; c < cols; ++c) Mref(n + 1 + i, c) = sgn * Mref(n - 1 - i, c);
    for (int i = 0; i <= 2 * n - 2; ++i)
        for (int c = 0; c < cols; ++c) Mref(2 * n + 1 + i, c) = Mref(2 * n - 1 - i, c);

    // FFT each column (length 4n), take real parts of rows 0..2n.
    std::vector<double> Mr(static_cast<std::size_t>(2 * n + 1) * cols, 0.0);
    auto MrRef = [&](int r, int c) -> double& { return Mr[static_cast<std::size_t>(r) * cols + c]; };
    std::vector<double> col(rows);
    for (int c = 0; c < cols; ++c) {
        for (int r = 0; r < rows; ++r) col[r] = Mref(r, c);
        const std::vector<cd> out = fftFwdReal(col);
        for (int r = 0; r <= 2 * n; ++r) MrRef(r, c) = out[r].real();
    }
    for (int r = 1; r <= 2 * n - 1; ++r)
        for (int c = 0; c < cols; ++c) MrRef(r, c) *= 2.0;
    const double scale = 1.0 / static_cast<double>(2 * dd);
    for (int r = 0; r <= 2 * n; ++r)
        for (int c = 0; c < cols; ++c) MrRef(r, c) *= scale;

    F.assign(n, 0.0);
    dF.assign(static_cast<std::size_t>(n) * n, 0.0);
    for (int i = 0; i < n; ++i) {
        const int row = parity + 2 * i;
        F[i] = MrRef(row, n);
        for (int j = 0; j < n; ++j) dF[static_cast<std::size_t>(i) * n + j] = MrRef(row, j);
    }
}

std::vector<double> newtonSolve(const std::vector<double>& coef, int parity,
                                double crit, int maxiter, int& iters)
{
    const int n = static_cast<int>(coef.size());
    std::vector<double> red(n);
    for (int i = 0; i < n; ++i) red[i] = coef[i] / 2.0;
    std::vector<double> F, dF;
    iters = 0;
    for (int it = 0; it < maxiter; ++it) {
        genJacobian(red, parity, F, dF);
        std::vector<double> res(n);
        double err = 0.0;
        for (int i = 0; i < n; ++i) {
            res[i] = F[i] - coef[i];
            err += std::abs(res[i]); // l1 norm
        }
        const std::vector<double> step = solveLinear(dF, res, n);
        for (int i = 0; i < n; ++i) red[i] -= step[i];
        ++iters;
        if (err < crit) break;
    }
    return red;
}

std::vector<double> reconstructFull(const std::vector<double>& red, int parity)
{
    const int n = static_cast<int>(red.size());
    std::vector<double> full;
    if (parity == 1) {
        for (int k = n - 1; k >= 0; --k) full.push_back(red[k]);
        for (int k = 0; k < n; ++k) full.push_back(red[k]);
    } else {
        if (n == 1) {
            full.push_back(2 * red[0]);
        } else {
            for (int k = n - 1; k >= 1; --k) full.push_back(red[k]);
            full.push_back(2 * red[0]);
            for (int k = 1; k < n; ++k) full.push_back(red[k]);
        }
    }
    // Im-protocol -> Re-protocol adapter.
    full.front() -= kPi / 4;
    full.back() -= kPi / 4;
    return full;
}

// Reduced Chebyshev coeffs of a degree-d target callable (parity d%2).
std::vector<double> chebReduced(const std::function<double(double)>& f, int d)
{
    const int N = d + 1, parity = d % 2, n = d / 2 + 1;
    std::vector<double> c(d + 1, 0.0);
    // Evaluate the target once per Chebyshev node. This used to sit inside the
    // k loop, costing (d+1)^2 calls to f -- and for a Chebyshev-series target
    // (what the C ABI builds) each call is itself O(d), making the transform
    // O(d^3): degree 1001 took ~78 s through qsp_solve_chebyshev versus ~3 s
    // through a C++ callable. Hoisting makes both paths O(d^2).
    std::vector<double> th(N), fv(N);
    for (int j = 0; j < N; ++j) {
        th[j] = kPi * (j + 0.5) / N;
        fv[j] = f(std::cos(th[j]));
    }
    for (int k = 0; k <= d; ++k) {
        double sum = 0.0;
        for (int j = 0; j < N; ++j) sum += fv[j] * std::cos(k * th[j]);
        c[k] = (k == 0 ? 1.0 / N : 2.0 / N) * sum;
    }
    std::vector<double> red(n);
    for (int i = 0; i < n; ++i) red[i] = c[parity + 2 * i];
    return red;
}

} // namespace

double SymQspAngleSolver::response(double x, const std::vector<double>& phases)
{
    const cd I(0.0, 1.0);
    const double s = std::sqrt(std::max(0.0, 1.0 - x * x));
    // W(x) = [[x, i s],[i s, x]].
    const Mat2c W{cd(x, 0), cd(0, s), cd(0, s), cd(x, 0)};
    auto ephiZ = [&](double phi) {
        return Mat2c{std::exp(I * phi), cd(0, 0), cd(0, 0), std::exp(-I * phi)};
    };
    Mat2c U = ephiZ(phases[0]);
    for (std::size_t k = 1; k < phases.size(); ++k) {
        U = mul2(mul2(U, W), ephiZ(phases[k]));
    }
    return U[0].real();
}

QspSolveResult SymQspAngleSolver::solve(
    const std::function<double(double)>& target) const
{
    const int d = degree_;
    const int parity = d % 2;

    int iters = 0;
    const std::vector<double> coef = chebReduced(target, d);
    const std::vector<double> red = newtonSolve(coef, parity, kNewtonCrit, kNewtonMaxIter, iters);

    QspSolveResult result;
    result.phases = reconstructFull(red, parity);
    result.iterations = iters;

    // Report the worst-case error over a fine grid on [-1, 1], measured with this
    // core's own self-contained response (Re<0|U|0>), so the residual is directly
    // comparable to the homotopy solver's.
    double worst = 0.0;
    const int grid = 401;
    for (int i = 0; i < grid; ++i) {
        const double x = -1.0 + 2.0 * i / (grid - 1);
        const double err =
            std::abs(SymQspAngleSolver::response(x, result.phases) - target(x));
        // std::max(a, NaN) returns a, so a non-finite sample would be silently
        // dropped and a diverged solve would report residual 0 / converged.
        if (!std::isfinite(err)) {
            result.residual = NAN;
            result.converged = false;
            return result;
        }
        worst = std::max(worst, err);
    }
    result.residual = worst;
    result.converged = worst < 1e-6;
    return result;
}

} // namespace qsvt
