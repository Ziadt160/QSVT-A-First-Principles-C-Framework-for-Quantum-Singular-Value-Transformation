#include "LcuBlockEncoding.hpp"

#include <cmath>
#include <complex>
#include <stdexcept>
#include <vector>

namespace qsvt {

namespace {

using Eigen::Matrix2cd;

constexpr double kPi = 3.14159265358979323846;

// ---- single-qubit building blocks ------------------------------------------

Matrix2cd ryMatrix(double theta)
{
    Matrix2cd m;
    const double c = std::cos(theta / 2.0);
    const double s = std::sin(theta / 2.0);
    m << c, -s,
         s, c;
    return m;
}

Matrix2cd pauliMatrix(char p)
{
    Matrix2cd m;
    switch (p) {
    case 'I':
        m << 1, 0, 0, 1;
        break;
    case 'X':
        m << 0, 1, 1, 0;
        break;
    case 'Y':
        m << Complex(0, 0), Complex(0, -1), Complex(0, 1), Complex(0, 0);
        break;
    case 'Z':
        m << 1, 0, 0, -1;
        break;
    default:
        throw std::invalid_argument(std::string("invalid Pauli char: ") + p);
    }
    return m;
}

Matrix2cd xGate() { return pauliMatrix('X'); }

// ---- multi-controlled single-qubit gate, synthesised to {single, CNOT} ------
//
// We build C^k(U): apply the 2x2 unitary U to `target` iff every qubit in
// `controls` is |1>. Strategy (correctness-first, poly(n)):
//   * compute the AND of all controls into a fresh work qubit via a ladder of
//     Toffolis, each Toffoli expanded into the standard 6-CNOT + T/H Clifford+T
//     decomposition;
//   * apply a singly-controlled-U (control = the AND work qubit) via the
//     Nielsen-Chuang ABC + 2-CNOT construction;
//   * uncompute the ladder, returning all work qubits to |0>.
// Work qubits are appended above the highest used index; the caller tells us
// where the first free work qubit lives via `firstWork`. The number of work
// qubits needed for k controls is max(0, k-1); they are all returned to |0>.

// Standard Toffoli(c0, c1 -> t) in {H, T, Tdg, S, CNOT}. Returns the gate list.
void appendToffoli(std::vector<Gate>& g, int c0, int c1, int t)
{
    // T and its adjoint, S, H as 2x2 matrices.
    Matrix2cd H;
    H << 1, 1, 1, -1;
    H *= (1.0 / std::sqrt(2.0));

    Matrix2cd T;
    T << 1, 0, 0, std::exp(Complex(0, kPi / 4.0));
    Matrix2cd Tdg;
    Tdg << 1, 0, 0, std::exp(Complex(0, -kPi / 4.0));

    // Sleator-Weinfurter / standard Toffoli decomposition.
    g.push_back(Gate::single(H, t));
    g.push_back(Gate::cnot(c1, t));
    g.push_back(Gate::single(Tdg, t));
    g.push_back(Gate::cnot(c0, t));
    g.push_back(Gate::single(T, t));
    g.push_back(Gate::cnot(c1, t));
    g.push_back(Gate::single(Tdg, t));
    g.push_back(Gate::cnot(c0, t));
    g.push_back(Gate::single(T, c1));
    g.push_back(Gate::single(T, t));
    g.push_back(Gate::single(H, t));
    g.push_back(Gate::cnot(c0, c1));
    g.push_back(Gate::single(T, c0));
    g.push_back(Gate::single(Tdg, c1));
    g.push_back(Gate::cnot(c0, c1));
}

// Rz(t) = diag(e^{-i t/2}, e^{i t/2}).
Matrix2cd rzMatrix(double t)
{
    Matrix2cd m;
    m << std::exp(Complex(0, -t / 2.0)), 0,
         0, std::exp(Complex(0, t / 2.0));
    return m;
}

// Singly-controlled-U: apply 2x2 U to `target` iff `control` is |1>.
// Nielsen-Chuang 4.3: with U = e^{i alpha} A X B X C, ABC = I, the controlled
// gate is  A.(CNOT).B.(CNOT).C  on the target plus diag(1, e^{i alpha}) on the
// control. We obtain A, B, C from a numerically robust ZYZ decomposition that
// first factors the global phase out via the determinant, so V = e^{-i alpha} U
// lies in SU(2) and its ZYZ angles are well conditioned in every case
// (diagonal, antidiagonal, or general).
void appendControlledU(std::vector<Gate>& g, const Matrix2cd& U, int control,
                       int target)
{
    // Global phase: det(U) = e^{2 i alpha}.  V = e^{-i alpha} U in SU(2).
    const Complex det = U(0, 0) * U(1, 1) - U(0, 1) * U(1, 0);
    const double alpha = 0.5 * std::arg(det);
    const Matrix2cd V = std::exp(Complex(0, -alpha)) * U;

    // ZYZ of V in SU(2):  V = Rz(beta) Ry(gamma) Rz(delta).
    //   |V00| = cos(gamma/2),  |V10| = sin(gamma/2)
    //   arg(V11) - arg(V00) = beta + delta   (from the diagonal)
    //   arg(V10) - arg(V00) = (beta - delta)/... handled via the two phases.
    const double gamma = 2.0 * std::atan2(std::abs(V(1, 0)), std::abs(V(0, 0)));

    double beta, delta;
    const bool haveOff = std::abs(V(1, 0)) > 1e-12 && std::abs(V(0, 0)) > 1e-12;
    if (haveOff) {
        // V00 = cos(g/2) e^{-i(beta+delta)/2}, V10 = sin(g/2) e^{ i(beta-delta)/2}
        const double pBeta = -std::arg(V(0, 0));   // (beta+delta)/2
        const double mDelta = std::arg(V(1, 0));   // (beta-delta)/2
        beta = pBeta + mDelta;
        delta = pBeta - mDelta;
    } else if (std::abs(V(0, 0)) > 1e-12) {
        // diagonal: V = diag(e^{-i(beta+delta)/2}, e^{i(beta+delta)/2})
        beta = -2.0 * std::arg(V(0, 0));
        delta = 0.0;
    } else {
        // antidiagonal: V10 = sin(g/2) e^{i(beta-delta)/2}, gamma = pi
        beta = 2.0 * std::arg(V(1, 0));
        delta = 0.0;
    }

    // A = Rz(beta) Ry(gamma/2)
    // B = Ry(-gamma/2) Rz(-(delta+beta)/2)
    // C = Rz((delta-beta)/2)
    // Then A X B X C = V (with X = sigma_x), and ABC = I.
    const Matrix2cd A = rzMatrix(beta) * ryMatrix(gamma / 2.0);
    const Matrix2cd B = ryMatrix(-gamma / 2.0) * rzMatrix(-(delta + beta) / 2.0);
    const Matrix2cd C = rzMatrix((delta - beta) / 2.0);

    // Controlled phase e^{i alpha} on the control line: diag(1, e^{i alpha}).
    Matrix2cd phase;
    phase << 1, 0, 0, std::exp(Complex(0, alpha));

    // C-then-X-then-B-then-X-then-A, with the X's being CNOTs from control.
    g.push_back(Gate::single(C, target));
    g.push_back(Gate::cnot(control, target));
    g.push_back(Gate::single(B, target));
    g.push_back(Gate::cnot(control, target));
    g.push_back(Gate::single(A, target));
    g.push_back(Gate::single(phase, control));
}

// Multi-controlled-U: apply 2x2 U to `target` iff all `controls` are |1>.
// `work` lists free |0> work qubits (>= controls.size()-1 of them needed).
// All work qubits are returned to |0>.
void appendMultiControlledU(std::vector<Gate>& g, const Matrix2cd& U,
                            const std::vector<int>& controls, int target,
                            const std::vector<int>& work)
{
    const int k = static_cast<int>(controls.size());
    if (k == 0) {
        g.push_back(Gate::single(U, target));
        return;
    }
    if (k == 1) {
        appendControlledU(g, U, controls[0], target);
        return;
    }

    // Compute AND of all controls into work[0..k-2] via a Toffoli ladder.
    //   work[0] = controls[0] AND controls[1]
    //   work[i] = work[i-1] AND controls[i+1]   for i = 1..k-2
    // Then a singly-controlled-U on work[k-2], then uncompute the ladder.
    if (static_cast<int>(work.size()) < k - 1) {
        throw std::logic_error("appendMultiControlledU: not enough work qubits");
    }

    std::vector<Gate> ladder;
    appendToffoli(ladder, controls[0], controls[1], work[0]);
    for (int i = 1; i <= k - 2; ++i) {
        appendToffoli(ladder, work[i - 1], controls[i + 1], work[i]);
    }

    // forward ladder
    for (const Gate& gg : ladder) {
        g.push_back(gg);
    }
    // controlled-U on the final AND wire
    appendControlledU(g, U, work[k - 2], target);
    // uncompute ladder (reverse order; Toffoli blocks are self-inverse as a
    // whole, so reversing the gate order undoes them).
    for (auto it = ladder.rbegin(); it != ladder.rend(); ++it) {
        Gate gg = *it;
        if (!gg.isCnot) {
            gg.matrix = gg.matrix.adjoint().eval();
        }
        g.push_back(gg);
    }
}

// ---- PREPARE: Mottonen real-amplitude state prep ---------------------------
//
// Build a circuit on `m` qubits (logical indices given by `qubits`, qubit
// qubits[0] = least-significant bit of the basis index k) that maps
//   |0> -> sum_k amp[k] |k>,   amp[k] >= 0,  sum amp[k]^2 = 1.
// Uses the recursive uniformly-controlled-Ry construction. The returned gates
// realise the map; only {single, CNOT} are used (uniformly-controlled Ry
// compiles to Ry + CNOT via the Gray-code / Walsh-Hadamard trick).

// Apply a uniformly-controlled Ry on `targetQ`, controlled by `controlQs`
// (controlQs are the higher-significance qubits). `angles` has length
// 2^{#controls}, indexed by the control bitstring (controlQs[0] = LSB of that
// index). Compiles to Ry rotations interleaved with CNOTs (Gray-code trick).
void appendUniformlyControlledRy(std::vector<Gate>& g,
                                 const std::vector<double>& angles,
                                 const std::vector<int>& controlQs, int targetQ)
{
    const int nc = static_cast<int>(controlQs.size());
    const int N = 1 << nc;

    if (nc == 0) {
        g.push_back(Gate::single(ryMatrix(angles[0]), targetQ));
        return;
    }

    // Transform the angles by the Walsh-Hadamard / "M" matrix so that emitting
    // Ry(theta'_j) then a CNOT (control = Gray-code bit) realises the uniformly
    // controlled rotation. Standard Mottonen formulation:
    //   theta'_i = (1/N) sum_j  (-1)^{ b(g(i)) . b(j) } theta_j
    // where g(i) is the i-th Gray code and b is the bit vector.
    auto grayCode = [](int x) { return x ^ (x >> 1); };
    auto parityDot = [](int a, int b) {
        unsigned v = static_cast<unsigned>(a & b);
        int p = 0;
        while (v) {
            p ^= 1;
            v &= v - 1;
        }
        return p;
    };

    std::vector<double> theta(N, 0.0);
    for (int i = 0; i < N; ++i) {
        double acc = 0.0;
        const int gi = grayCode(i);
        for (int j = 0; j < N; ++j) {
            const double sign = parityDot(gi, j) ? -1.0 : 1.0;
            acc += sign * angles[j];
        }
        theta[i] = acc / static_cast<double>(N);
    }

    // Emit: for i in 0..N-1: Ry(theta[i]) on target, then CNOT from the control
    // qubit whose Gray-code bit flips between step i and i+1.
    for (int i = 0; i < N; ++i) {
        g.push_back(Gate::single(ryMatrix(theta[i]), targetQ));
        // bit that changes from gray(i) to gray(i+1); for the last step we wrap
        // to close the cycle (flip the highest control bit so the net CNOT
        // pattern is the identity on the control register).
        int ctrlBit;
        if (i == N - 1) {
            ctrlBit = nc - 1;
        } else {
            const int flip = grayCode(i) ^ grayCode(i + 1);
            // flip is a single set bit
            ctrlBit = 0;
            int f = flip;
            while ((f & 1) == 0) {
                f >>= 1;
                ++ctrlBit;
            }
        }
        g.push_back(Gate::cnot(controlQs[ctrlBit], targetQ));
    }
}

// Recursively build the state-prep for amplitudes `amp` (length 2^m) on
// `qubits` (qubits[0] = LSB of k). qubits[m-1] is the most-significant; we
// rotate it first, then recurse into each of its two halves with a uniform
// control on it.
void appendStatePrep(std::vector<Gate>& g, const std::vector<double>& amp,
                     const std::vector<int>& qubits)
{
    const int m = static_cast<int>(qubits.size());
    if (m == 0) {
        return;
    }

    // The most significant qubit is qubits[m-1]. Splitting k by that bit:
    //   lower half  (bit = 0): amp[0 .. half-1]
    //   upper half  (bit = 1): amp[half .. 2*half-1]
    const int half = 1 << (m - 1);

    // For each value of the lower (m-1) qubits, the MSB rotation angle is
    //   theta = 2 * atan2( norm(upper branch), norm(lower branch) )
    // i.e. a uniformly-controlled Ry on qubits[m-1] controlled by qubits[0..m-2].
    std::vector<double> angles(half, 0.0);
    std::vector<double> lowerAmp(half, 0.0); // marginal over MSB for recursion
    for (int j = 0; j < half; ++j) {
        const double a0 = amp[j];        // MSB = 0
        const double a1 = amp[j + half]; // MSB = 1
        const double norm = std::sqrt(a0 * a0 + a1 * a1);
        lowerAmp[j] = norm;
        if (norm < 1e-300) {
            angles[j] = 0.0;
        } else {
            angles[j] = 2.0 * std::atan2(a1, a0);
        }
    }

    // First prepare the (m-1)-qubit marginal distribution on qubits[0..m-2],
    // then apply the uniformly-controlled rotation that "opens up" the MSB.
    std::vector<int> lowerQubits(qubits.begin(), qubits.end() - 1);
    appendStatePrep(g, lowerAmp, lowerQubits);
    appendUniformlyControlledRy(g, angles, lowerQubits, qubits[m - 1]);
}

} // namespace

LcuBlockEncoding::LcuBlockEncoding(const std::vector<PauliTerm>& terms,
                                   int n_system)
    : terms_(terms), n_system_(n_system)
{
    if (n_system <= 0) {
        throw std::invalid_argument("n_system must be positive");
    }
    const int L = static_cast<int>(terms_.size());
    if (L == 0) {
        throw std::invalid_argument("need at least one term");
    }
    for (const PauliTerm& t : terms_) {
        if (static_cast<int>(t.pauli.size()) != n_system) {
            throw std::invalid_argument(
                "pauli string length must equal n_system");
        }
        if (std::abs(t.coeff) == 0.0) {
            throw std::invalid_argument("zero coefficient not allowed");
        }
    }

    // m = ceil(log2 L) ancilla qubits for the PREPARE register.
    n_ancilla_ = 0;
    while ((1 << n_ancilla_) < L) {
        ++n_ancilla_;
    }
    const int m = n_ancilla_;
    const int M = 1 << m;

    // alpha = sum |c_k|, and the PREPARE amplitudes amp[k] = sqrt(|c_k|/alpha).
    alpha_ = 0.0;
    for (const PauliTerm& t : terms_) {
        alpha_ += std::abs(t.coeff);
    }
    std::vector<double> amp(M, 0.0);
    for (int k = 0; k < L; ++k) {
        amp[k] = std::sqrt(std::abs(terms_[k].coeff) / alpha_);
    }

    // Qubit indices.
    std::vector<int> ancQ(m);
    for (int i = 0; i < m; ++i) {
        ancQ[i] = n_system_ + i; // ancilla are the HIGH qubits
    }
    // Work qubits for the multi-controlled gates: need up to m-1 of them.
    // Place them above the ancilla register. Note: these MUST be returned to
    // |0> after every multi-controlled gate (they are -- the ladder uncomputes).
    const int nWork = (m >= 2) ? (m - 1) : 0;
    std::vector<int> workQ(nWork);
    for (int i = 0; i < nWork; ++i) {
        workQ[i] = n_system_ + m + i;
    }

    // ---- PREPARE -----------------------------------------------------------
    std::vector<Gate> prep;
    appendStatePrep(prep, amp, ancQ);

    // ---- SELECT ------------------------------------------------------------
    std::vector<Gate> select;
    for (int k = 0; k < L; ++k) {
        const PauliTerm& t = terms_[k];
        const Complex phase = t.coeff / std::abs(t.coeff); // c_k / |c_k|

        // X gates on the ancilla qubits whose bit in k is 0, so the all-ones
        // control pattern selects |k>.
        std::vector<Gate> xMask;
        for (int b = 0; b < m; ++b) {
            if (((k >> b) & 1) == 0) {
                xMask.push_back(Gate::single(xGate(), ancQ[b]));
            }
        }
        for (const Gate& gg : xMask) {
            select.push_back(gg);
        }

        // Apply U_k = phase * P_k as m-controlled single-qubit gates. Fold the
        // global phase into the FIRST non-identity factor; if P_k is all-I,
        // apply the phase as a controlled global phase (diag(1,1)*phase) on any
        // system qubit -- realised as a controlled-phase gate on the target.
        // Pauli char at string position i acts on system qubit (n_system-1-i).
        bool phaseFolded = false;
        // First collect non-identity factors.
        std::vector<std::pair<int, char>> factors; // (system qubit, pauli char)
        for (int i = 0; i < n_system_; ++i) {
            const char p = t.pauli[static_cast<std::size_t>(i)];
            if (p != 'I') {
                factors.emplace_back(n_system_ - 1 - i, p);
            }
        }

        if (factors.empty()) {
            // P_k = identity: U_k = phase * I. Apply a controlled global phase
            // diag(phase, phase) -- equivalently a phase on the ancilla |k>
            // subspace. Realise as an m-controlled gate U = phase * I on any
            // system qubit (the matrix phase*I commutes through, giving the
            // overall scalar on the selected block).
            Matrix2cd pm;
            pm << phase, 0, 0, phase;
            std::vector<int> controls = ancQ;
            appendMultiControlledU(select, pm, controls, 0, workQ);
        } else {
            for (std::size_t fi = 0; fi < factors.size(); ++fi) {
                const int sysQ = factors[fi].first;
                const char p = factors[fi].second;
                Matrix2cd pm = pauliMatrix(p);
                if (!phaseFolded) {
                    pm = phase * pm; // fold sign/phase into the first factor
                    phaseFolded = true;
                }
                std::vector<int> controls = ancQ;
                appendMultiControlledU(select, pm, controls, sysQ, workQ);
            }
        }

        // Undo the X mask.
        for (const Gate& gg : xMask) {
            select.push_back(gg); // X is self-inverse
        }
    }

    // ---- PREPARE^dagger ----------------------------------------------------
    std::vector<Gate> prepDag;
    prepDag.reserve(prep.size());
    for (auto it = prep.rbegin(); it != prep.rend(); ++it) {
        Gate gg = *it;
        if (!gg.isCnot) {
            gg.matrix = gg.matrix.adjoint().eval();
        }
        prepDag.push_back(gg);
    }

    // ---- assemble: PREPARE, SELECT, PREPARE^dagger -------------------------
    gates_.clear();
    gates_.reserve(prep.size() + select.size() + prepDag.size());
    for (const Gate& gg : prep) {
        gates_.push_back(gg);
    }
    for (const Gate& gg : select) {
        gates_.push_back(gg);
    }
    for (const Gate& gg : prepDag) {
        gates_.push_back(gg);
    }
}

} // namespace qsvt
