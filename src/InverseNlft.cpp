#include "InverseNlft.hpp"

#include <cmath>
#include <stdexcept>

namespace qsvt {

std::vector<std::vector<Complexd>> halfCholeskyColumns(std::vector<Complexd> u,
                                                       std::vector<Complexd> v)
{
    if (u.size() != v.size() || u.empty()) {
        throw std::invalid_argument("halfCholeskyColumns: u and v must match and be non-empty");
    }
    const std::size_t n = u.size() - 1;

    std::vector<std::vector<Complexd>> cols;
    cols.reserve(n + 1);

    for (std::size_t k = 0; k < n; ++k) {
        const std::size_t m = u.size(); // = n + 1 - k
        const Complexd u0 = u[0];
        const Complexd v0 = v[0];
        const double nrm = std::sqrt(std::norm(u0) + std::norm(v0));
        if (nrm == 0.0) {
            throw std::runtime_error("halfCholeskyColumns: singular pivot");
        }

        // One Givens rotation applied to both rows (see header).
        std::vector<Complexd> up(m), vp(m);
        for (std::size_t j = 0; j < m; ++j) {
            up[j] = (std::conj(u0) * u[j] + std::conj(v0) * v[j]) / nrm;
            vp[j] = (-v0 * u[j] + u0 * v[j]) / nrm;
        }

        // Store the column, normalised by its leading entry (= nrm).
        std::vector<Complexd> col(m);
        for (std::size_t j = 0; j < m; ++j) col[j] = up[j] / up[0];
        cols.push_back(std::move(col));

        // Next block: (u'_0..u'_{m-2}, v'_1..v'_{m-1}).
        std::vector<Complexd> uNext(m - 1), vNext(m - 1);
        for (std::size_t j = 0; j + 1 < m; ++j) {
            uNext[j] = up[j];
            vNext[j] = vp[j + 1];
        }
        u.swap(uNext);
        v.swap(vNext);
    }

    cols.push_back({Complexd(1.0, 0.0)}); // last column
    return cols;
}

NlftSequence inverseNlft(const LaurentPoly& b, const LaurentPoly& c)
{
    const int n = b.effectiveDegree();
    const std::size_t len = static_cast<std::size_t>(n) + 1;

    // p = (conj(c[k])) for k running over b's support, highest exponent first.
    std::vector<Complexd> p(len, Complexd(0.0, 0.0));
    for (std::size_t i = 0; i < len; ++i) {
        const int k = b.supportEnd() - 1 - static_cast<int>(i);
        p[i] = std::conj(c.at(k));
    }

    std::vector<Complexd> e0(len, Complexd(0.0, 0.0));
    e0[0] = Complexd(1.0, 0.0);

    const std::vector<std::vector<Complexd>> cols = halfCholeskyColumns(e0, p);

    // Forward substitution: (F_n*, ..., F_0*) = L^{-1} p.
    // L[k][j] = cols[j][k - j] for j <= k.
    std::vector<Complexd> F(len, Complexd(0.0, 0.0));
    for (std::size_t k = 0; k < len; ++k) {
        Complexd acc = p[k];
        for (std::size_t j = 0; j < k; ++j) {
            const std::size_t idx = k - j;
            if (idx < cols[j].size()) acc -= cols[j][idx] * F[j];
        }
        F[k] = acc;
    }

    NlftSequence seq;
    seq.coeffs.resize(len);
    for (std::size_t i = 0; i < len; ++i) seq.coeffs[i] = std::conj(F[len - 1 - i]);
    seq.supportStart = b.supportStart;
    return seq;
}

} // namespace qsvt
