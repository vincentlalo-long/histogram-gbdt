// C++/OpenMP kernels for histogram GBDT.
//
// `find_best_split` is a standalone per-node split finder.  `build_tree`
// grows a complete level-wise tree in C++ and uses histogram subtraction:
// the smaller child's histogram is built from its rows, while the larger
// child is obtained by subtracting that histogram from its parent.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;

namespace {

struct BestSplit {
    int feature = -1;
    int bin = -1;
    double gain = 0.0;
    int64_t left_count = 0;
    double left_grad = 0.0;
    double left_hess = 0.0;
    bool default_left = false;
};

struct Histogram {
    std::vector<double> grad;
    std::vector<double> hess;
    std::vector<int64_t> count;
};

struct BuildNode {
    std::vector<int64_t> rows;
    int depth = 0;
    double grad_sum = 0.0;
    double hess_sum = 0.0;
    int64_t count = 0;
    Histogram hist;
};

struct OutputNode {
    int feature = -1;
    int bin = 0;
    int left = -1;
    int right = -1;
    double value = 0.0;
    double gain = 0.0;
    bool default_left = false;
};

int normalise_threads(int n_threads) {
#ifdef _OPENMP
    if (n_threads <= 0) {
        return omp_get_max_threads();
    }
    return n_threads;
#else
    (void)n_threads;
    return 1;
#endif
}

double leaf_value(double grad_sum, double hess_sum, double reg_lambda) {
    const double denom = hess_sum + reg_lambda;
    return denom > 0.0 ? -grad_sum / denom : 0.0;
}

BestSplit scan_feature(
    const double* hist_grad,
    const double* hist_hess,
    const int64_t* hist_count,
    int feature,
    int missing_bin,
    int n_bins,
    double total_grad,
    double total_hess,
    int64_t total_count,
    double parent_term,
    double reg_lambda,
    double gamma,
    double min_child_weight,
    int min_samples_leaf) {
    BestSplit best;
    best.feature = feature;

    double missing_grad = 0.0;
    double missing_hess = 0.0;
    int64_t missing_count = 0;
    int non_missing_limit = n_bins;
    if (missing_bin < n_bins) {
        missing_grad = hist_grad[missing_bin];
        missing_hess = hist_hess[missing_bin];
        missing_count = hist_count[missing_bin];
        non_missing_limit = missing_bin;
    }

    const double non_missing_grad = total_grad - missing_grad;
    const double non_missing_hess = total_hess - missing_hess;
    const int64_t non_missing_count = total_count - missing_count;

    double left_grad = 0.0;
    double left_hess = 0.0;
    int64_t left_count = 0;

    for (int bin = 0; bin < non_missing_limit - 1; ++bin) {
        left_grad += hist_grad[bin];
        left_hess += hist_hess[bin];
        left_count += hist_count[bin];

        const int64_t right_count = non_missing_count - left_count;
        if (left_count <= 0 || right_count <= 0) {
            continue;
        }

        const double right_grad = non_missing_grad - left_grad;
        const double right_hess = non_missing_hess - left_hess;

        // Missing values go to the right child.  This candidate is
        // evaluated first, so equal-gain ties retain right routing.
        {
            const int64_t left_child_count = left_count;
            const int64_t right_child_count = right_count + missing_count;
            const double left_child_grad = left_grad;
            const double left_child_hess = left_hess;
            const double right_child_grad = right_grad + missing_grad;
            const double right_child_hess = right_hess + missing_hess;
            const double left_denom = left_child_hess + reg_lambda;
            const double right_denom = right_child_hess + reg_lambda;
            if (left_child_count >= min_samples_leaf &&
                right_child_count >= min_samples_leaf &&
                left_child_hess >= min_child_weight &&
                right_child_hess >= min_child_weight &&
                left_denom > 0.0 && right_denom > 0.0) {
                const double gain =
                    0.5 * (left_child_grad * left_child_grad / left_denom +
                           right_child_grad * right_child_grad / right_denom -
                           parent_term) -
                    gamma;
                if (gain > best.gain +
                                1e-12 * std::max(1.0, std::abs(best.gain))) {
                    best.gain = gain;
                    best.bin = bin;
                    best.left_count = left_child_count;
                    best.left_grad = left_child_grad;
                    best.left_hess = left_child_hess;
                    best.default_left = false;
                }
            }
        }

        // Missing values go to the left child.
        {
            const int64_t left_child_count = left_count + missing_count;
            const int64_t right_child_count = right_count;
            const double left_child_grad = left_grad + missing_grad;
            const double left_child_hess = left_hess + missing_hess;
            const double right_child_grad = right_grad;
            const double right_child_hess = right_hess;
            const double left_denom = left_child_hess + reg_lambda;
            const double right_denom = right_child_hess + reg_lambda;
            if (left_child_count >= min_samples_leaf &&
                right_child_count >= min_samples_leaf &&
                left_child_hess >= min_child_weight &&
                right_child_hess >= min_child_weight &&
                left_denom > 0.0 && right_denom > 0.0) {
                const double gain =
                    0.5 * (left_child_grad * left_child_grad / left_denom +
                           right_child_grad * right_child_grad / right_denom -
                           parent_term) -
                    gamma;
                if (gain > best.gain +
                                1e-12 * std::max(1.0, std::abs(best.gain))) {
                    best.gain = gain;
                    best.bin = bin;
                    best.left_count = left_child_count;
                    best.left_grad = left_child_grad;
                    best.left_hess = left_child_hess;
                    best.default_left = true;
                }
            }
        }
    }

    return best;
}

py::tuple version_split(
    int feature,
    int bin,
    double gain,
    int64_t left_count,
    double left_grad,
    double left_hess,
    bool default_left) {
    return py::make_tuple(feature, bin, gain, static_cast<long long>(left_count),
                          left_grad, left_hess, default_left);
}

py::tuple find_best_split(
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> X_,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> rows_,
    py::array_t<double, py::array::c_style | py::array::forcecast> grad_,
    py::array_t<double, py::array::c_style | py::array::forcecast> hess_,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> features_,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> missing_bins_,
    int n_bins,
    double reg_lambda,
    double gamma,
    double min_child_weight,
    int min_samples_leaf,
    int n_threads) {
    if (X_.ndim() != 2) {
        throw std::invalid_argument("Xbin must be a 2-D (n_features, n_samples) array");
    }
    if (rows_.ndim() != 1 || grad_.ndim() != 1 || hess_.ndim() != 1 ||
        features_.ndim() != 1 || missing_bins_.ndim() != 1) {
        throw std::invalid_argument(
            "rows, grad, hess, features, and missing_bins must be 1-D arrays");
    }
    if (n_bins < 1 || n_bins > 256) {
        throw std::invalid_argument("n_bins must be in [1, 256]");
    }
    if (min_samples_leaf < 1) {
        throw std::invalid_argument("min_samples_leaf must be at least 1");
    }

    const int n_features = static_cast<int>(X_.shape(0));
    const int n_samples = static_cast<int>(X_.shape(1));
    const int n_rows = static_cast<int>(rows_.shape(0));
    const int n_candidates = static_cast<int>(features_.shape(0));

    if (grad_.shape(0) != n_samples || hess_.shape(0) != n_samples) {
        throw std::invalid_argument("grad and hess must have length n_samples");
    }
    if (missing_bins_.shape(0) != n_features) {
        throw std::invalid_argument("missing_bins must have length n_features");
    }
    if (n_rows == 0 || n_candidates == 0) {
        return version_split(-1, -1,
                             -std::numeric_limits<double>::infinity(),
                             0, 0.0, 0.0, false);
    }

    const uint8_t* X = X_.data();
    const int64_t* rows = rows_.data();
    const double* grad = grad_.data();
    const double* hess = hess_.data();
    const int64_t* features = features_.data();
    const int64_t* missing_bins = missing_bins_.data();

    for (int ci = 0; ci < n_candidates; ++ci) {
        const int64_t f = features[ci];
        if (f < 0 || f >= n_features) {
            throw std::invalid_argument("feature index out of bounds");
        }
        const int64_t mb = missing_bins[f];
        if (mb < 0 || mb > n_bins) {
            throw std::invalid_argument("missing bin index out of bounds");
        }
    }
    for (int ri = 0; ri < n_rows; ++ri) {
        const int64_t r = rows[ri];
        if (r < 0 || r >= n_samples) {
            throw std::invalid_argument("row index out of bounds");
        }
    }

    double total_grad = 0.0;
    double total_hess = 0.0;
    for (int ri = 0; ri < n_rows; ++ri) {
        const int64_t r = rows[ri];
        total_grad += grad[r];
        total_hess += hess[r];
    }
    const double parent_denom = total_hess + reg_lambda;
    if (parent_denom <= 0.0) {
        return version_split(-1, -1,
                             -std::numeric_limits<double>::infinity(),
                             0, 0.0, 0.0, false);
    }
    const double parent_term = total_grad * total_grad / parent_denom;
    const int threads = normalise_threads(n_threads);

    std::vector<BestSplit> candidates(static_cast<size_t>(n_candidates));

    {
        py::gil_scoped_release release;

#pragma omp parallel for schedule(static) num_threads(threads)
        for (int ci = 0; ci < n_candidates; ++ci) {
            const int feature = static_cast<int>(features[ci]);
            std::vector<double> hist_grad(static_cast<size_t>(n_bins), 0.0);
            std::vector<double> hist_hess(static_cast<size_t>(n_bins), 0.0);
            std::vector<int64_t> hist_count(static_cast<size_t>(n_bins), 0);

            for (int ri = 0; ri < n_rows; ++ri) {
                const int64_t row = rows[ri];
                const int bin = static_cast<int>(
                    X[static_cast<size_t>(feature) * n_samples + row]);
                hist_grad[bin] += grad[row];
                hist_hess[bin] += hess[row];
                hist_count[bin] += 1;
            }

            candidates[static_cast<size_t>(ci)] = scan_feature(
                hist_grad.data(), hist_hess.data(), hist_count.data(),
                feature, static_cast<int>(missing_bins[feature]), n_bins,
                total_grad, total_hess, n_rows, parent_term, reg_lambda,
                gamma, min_child_weight, min_samples_leaf);
        }
    }

    BestSplit best;
    for (int ci = 0; ci < n_candidates; ++ci) {
        const BestSplit& candidate = candidates[static_cast<size_t>(ci)];
        if (candidate.bin >= 0 &&
            candidate.gain > best.gain +
                                1e-12 * std::max(1.0, std::abs(best.gain))) {
            best = candidate;
        }
    }

    if (best.bin < 0 || best.gain <= 0.0) {
        return version_split(-1, -1,
                             -std::numeric_limits<double>::infinity(),
                             0, 0.0, 0.0, false);
    }
    return version_split(best.feature, best.bin, best.gain, best.left_count,
                         best.left_grad, best.left_hess, best.default_left);
}

void build_histogram(
    const uint8_t* X,
    int n_samples,
    const std::vector<int64_t>& rows,
    const int64_t* features,
    int n_candidates,
    int n_bins,
    const double* grad,
    const double* hess,
    int n_threads,
    Histogram& hist) {
    const size_t width = static_cast<size_t>(n_candidates) * n_bins;
    hist.grad.assign(width, 0.0);
    hist.hess.assign(width, 0.0);
    hist.count.assign(width, 0);

    const int threads = normalise_threads(n_threads);
    const int n_rows = static_cast<int>(rows.size());

#pragma omp parallel for schedule(static) num_threads(threads)
    for (int ci = 0; ci < n_candidates; ++ci) {
        const int feature = static_cast<int>(features[ci]);
        const size_t offset = static_cast<size_t>(ci) * n_bins;
        for (int ri = 0; ri < n_rows; ++ri) {
            const int64_t row = rows[ri];
            const int bin = static_cast<int>(
                X[static_cast<size_t>(feature) * n_samples + row]);
            hist.grad[offset + bin] += grad[row];
            hist.hess[offset + bin] += hess[row];
            hist.count[offset + bin] += 1;
        }
    }
}

void subtract_histogram(
    const Histogram& parent,
    const Histogram& child,
    int n_candidates,
    int n_bins,
    int n_threads,
    Histogram& result) {
    const size_t width = static_cast<size_t>(n_candidates) * n_bins;
    result.grad.resize(width);
    result.hess.resize(width);
    result.count.resize(width);

    const int threads = normalise_threads(n_threads);
#pragma omp parallel for schedule(static) num_threads(threads)
    for (int i = 0; i < static_cast<int>(width); ++i) {
        result.grad[i] = parent.grad[i] - child.grad[i];
        result.hess[i] = parent.hess[i] - child.hess[i];
        result.count[i] = parent.count[i] - child.count[i];
    }
}

BestSplit find_best_split_from_hist(
    const Histogram& hist,
    const int64_t* features,
    int n_candidates,
    const int64_t* missing_bins,
    int n_bins,
    double total_grad,
    double total_hess,
    int64_t total_count,
    double reg_lambda,
    double gamma,
    double min_child_weight,
    int min_samples_leaf,
    int n_threads) {
    const double parent_denom = total_hess + reg_lambda;
    if (parent_denom <= 0.0 || n_candidates == 0) {
        return BestSplit();
    }
    const double parent_term = total_grad * total_grad / parent_denom;
    const int threads = normalise_threads(n_threads);
    std::vector<BestSplit> candidates(static_cast<size_t>(n_candidates));

#pragma omp parallel for schedule(static) num_threads(threads)
    for (int ci = 0; ci < n_candidates; ++ci) {
        const int feature = static_cast<int>(features[ci]);
        const size_t offset = static_cast<size_t>(ci) * n_bins;
        candidates[static_cast<size_t>(ci)] = scan_feature(
            hist.grad.data() + offset,
            hist.hess.data() + offset,
            hist.count.data() + offset,
            feature,
            static_cast<int>(missing_bins[feature]),
            n_bins,
            total_grad,
            total_hess,
            total_count,
            parent_term,
            reg_lambda,
            gamma,
            min_child_weight,
            min_samples_leaf);
    }

    BestSplit best;
    for (int ci = 0; ci < n_candidates; ++ci) {
        const BestSplit& candidate = candidates[static_cast<size_t>(ci)];
        if (candidate.bin >= 0 &&
            candidate.gain > best.gain +
                                1e-12 * std::max(1.0, std::abs(best.gain))) {
            best = candidate;
        }
    }
    return best;
}

py::tuple build_tree(
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> X_,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> rows_,
    py::array_t<double, py::array::c_style | py::array::forcecast> grad_,
    py::array_t<double, py::array::c_style | py::array::forcecast> hess_,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> features_,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> missing_bins_,
    int n_bins,
    int max_depth,
    double reg_lambda,
    double gamma,
    double min_child_weight,
    int min_samples_leaf,
    int n_threads) {
    if (X_.ndim() != 2) {
        throw std::invalid_argument("Xbin must be a 2-D (n_features, n_samples) array");
    }
    if (rows_.ndim() != 1 || grad_.ndim() != 1 || hess_.ndim() != 1 ||
        features_.ndim() != 1 || missing_bins_.ndim() != 1) {
        throw std::invalid_argument(
            "rows, grad, hess, features, and missing_bins must be 1-D arrays");
    }
    if (n_bins < 1 || n_bins > 256) {
        throw std::invalid_argument("n_bins must be in [1, 256]");
    }
    if (max_depth < 0) {
        throw std::invalid_argument("max_depth must be non-negative");
    }
    if (min_samples_leaf < 1) {
        throw std::invalid_argument("min_samples_leaf must be at least 1");
    }

    const int n_features = static_cast<int>(X_.shape(0));
    const int n_samples = static_cast<int>(X_.shape(1));
    const int n_candidates = static_cast<int>(features_.shape(0));
    const int root_count = static_cast<int>(rows_.shape(0));

    if (grad_.shape(0) != n_samples || hess_.shape(0) != n_samples) {
        throw std::invalid_argument("grad and hess must have length n_samples");
    }
    if (missing_bins_.shape(0) != n_features) {
        throw std::invalid_argument("missing_bins must have length n_features");
    }
    if (root_count <= 0) {
        throw std::invalid_argument("cannot build a tree with no rows");
    }

    const uint8_t* X = X_.data();
    const double* grad = grad_.data();
    const double* hess = hess_.data();
    const int64_t* features = features_.data();
    const int64_t* missing_bins = missing_bins_.data();

    for (int ci = 0; ci < n_candidates; ++ci) {
        const int64_t f = features[ci];
        if (f < 0 || f >= n_features) {
            throw std::invalid_argument("feature index out of bounds");
        }
        const int64_t mb = missing_bins[f];
        if (mb < 0 || mb > n_bins) {
            throw std::invalid_argument("missing bin index out of bounds");
        }
    }
    const int64_t* root_rows = rows_.data();
    for (int ri = 0; ri < root_count; ++ri) {
        const int64_t r = root_rows[ri];
        if (r < 0 || r >= n_samples) {
            throw std::invalid_argument("row index out of bounds");
        }
    }

    std::vector<OutputNode> output;
    output.push_back(OutputNode());
    std::vector<BuildNode> nodes;

    double root_grad = 0.0;
    double root_hess = 0.0;
    for (int ri = 0; ri < root_count; ++ri) {
        const int64_t r = root_rows[ri];
        root_grad += grad[r];
        root_hess += hess[r];
    }

    {
        py::gil_scoped_release release;

        Histogram root_hist;
        build_histogram(X, n_samples, std::vector<int64_t>(root_rows, root_rows + root_count),
                        features, n_candidates, n_bins, grad, hess, n_threads,
                        root_hist);

        BuildNode root;
        root.rows.assign(root_rows, root_rows + root_count);
        root.depth = 0;
        root.grad_sum = root_grad;
        root.hess_sum = root_hess;
        root.count = root_count;
        root.hist = std::move(root_hist);
        nodes.push_back(std::move(root));

        size_t index = 0;
        while (index < nodes.size()) {
            BuildNode node = std::move(nodes[index]);

            bool make_leaf = false;
            if (node.depth >= max_depth ||
                node.count < min_samples_leaf ||
                node.hess_sum < min_child_weight) {
                make_leaf = true;
            }

            BestSplit split;
            if (!make_leaf) {
                split = find_best_split_from_hist(
                    node.hist, features, n_candidates, missing_bins, n_bins,
                    node.grad_sum, node.hess_sum, node.count, reg_lambda,
                    gamma, min_child_weight, min_samples_leaf, n_threads);
                if (split.bin < 0 || split.gain <= 0.0) {
                    make_leaf = true;
                }
            }

            if (make_leaf) {
                output[index].value = leaf_value(
                    node.grad_sum, node.hess_sum, reg_lambda);
                ++index;
                continue;
            }

            const int feature = split.feature;
            const int missing_bin = static_cast<int>(missing_bins[feature]);
            std::vector<int64_t> left_rows;
            std::vector<int64_t> right_rows;
            left_rows.reserve(static_cast<size_t>(node.count));
            right_rows.reserve(static_cast<size_t>(node.count));

            for (const int64_t row : node.rows) {
                const int bin = static_cast<int>(
                    X[static_cast<size_t>(feature) * n_samples + row]);
                bool go_left;
                if (bin == missing_bin) {
                    go_left = split.default_left;
                } else {
                    go_left = bin <= split.bin;
                }
                if (go_left) {
                    left_rows.push_back(row);
                } else {
                    right_rows.push_back(row);
                }
            }

            const int64_t left_count = static_cast<int64_t>(left_rows.size());
            const int64_t right_count = static_cast<int64_t>(right_rows.size());
            const double left_grad = split.left_grad;
            const double left_hess = split.left_hess;
            const double right_grad = node.grad_sum - left_grad;
            const double right_hess = node.hess_sum - left_hess;

            if (left_count < min_samples_leaf ||
                right_count < min_samples_leaf ||
                left_hess < min_child_weight ||
                right_hess < min_child_weight) {
                output[index].value = leaf_value(
                    node.grad_sum, node.hess_sum, reg_lambda);
                ++index;
                continue;
            }

            Histogram left_hist;
            Histogram right_hist;
            if (left_count <= right_count) {
                build_histogram(
                    X, n_samples, left_rows, features, n_candidates, n_bins,
                    grad, hess, n_threads, left_hist);
                subtract_histogram(
                    node.hist, left_hist, n_candidates, n_bins, n_threads,
                    right_hist);
            } else {
                build_histogram(
                    X, n_samples, right_rows, features, n_candidates, n_bins,
                    grad, hess, n_threads, right_hist);
                subtract_histogram(
                    node.hist, right_hist, n_candidates, n_bins, n_threads,
                    left_hist);
            }

            output[index].feature = feature;
            output[index].bin = split.bin;
            output[index].gain = split.gain;
            output[index].default_left = split.default_left;
            output[index].left = static_cast<int>(output.size());
            output[index].right = static_cast<int>(output.size()) + 1;
            output.push_back(OutputNode());
            output.push_back(OutputNode());

            BuildNode left_node;
            left_node.rows = std::move(left_rows);
            left_node.depth = node.depth + 1;
            left_node.grad_sum = left_grad;
            left_node.hess_sum = left_hess;
            left_node.count = left_count;
            left_node.hist = std::move(left_hist);
            nodes.push_back(std::move(left_node));

            BuildNode right_node;
            right_node.rows = std::move(right_rows);
            right_node.depth = node.depth + 1;
            right_node.grad_sum = right_grad;
            right_node.hess_sum = right_hess;
            right_node.count = right_count;
            right_node.hist = std::move(right_hist);
            nodes.push_back(std::move(right_node));

            ++index;
        }
    }

    const int n_nodes = static_cast<int>(output.size());
    py::array_t<int32_t> feature_array(n_nodes);
    py::array_t<int32_t> threshold_array(n_nodes);
    py::array_t<int32_t> left_array(n_nodes);
    py::array_t<int32_t> right_array(n_nodes);
    py::array_t<double> value_array(n_nodes);
    py::array_t<double> gain_array(n_nodes);
    py::array_t<uint8_t> default_left_array(n_nodes);

    int32_t* feature_out = feature_array.mutable_data();
    int32_t* threshold_out = threshold_array.mutable_data();
    int32_t* left_out = left_array.mutable_data();
    int32_t* right_out = right_array.mutable_data();
    double* value_out = value_array.mutable_data();
    double* gain_out = gain_array.mutable_data();
    uint8_t* default_left_out = default_left_array.mutable_data();

    for (int i = 0; i < n_nodes; ++i) {
        const OutputNode& node = output[static_cast<size_t>(i)];
        feature_out[i] = node.feature;
        threshold_out[i] = node.bin;
        left_out[i] = node.left;
        right_out[i] = node.right;
        value_out[i] = node.value;
        gain_out[i] = node.gain;
        default_left_out[i] = node.default_left ? 1 : 0;
    }

    return py::make_tuple(
        feature_array, threshold_array, left_array, right_array, value_array,
        gain_array, default_left_array);
}

}  // namespace

PYBIND11_MODULE(_core, m) {
    m.doc() = "OpenMP histogram split finder and tree builder for gbdt";
    m.def("find_best_split", &find_best_split,
          py::arg("Xbin"), py::arg("rows"), py::arg("grad"), py::arg("hess"),
          py::arg("features"), py::arg("missing_bins"), py::arg("n_bins"),
          py::arg("reg_lambda"), py::arg("gamma"),
          py::arg("min_child_weight"), py::arg("min_samples_leaf"),
          py::arg("n_threads") = -1,
          "Find the best histogram split for one node.");
    m.def("build_tree", &build_tree,
          py::arg("Xbin"), py::arg("rows"), py::arg("grad"), py::arg("hess"),
          py::arg("features"), py::arg("missing_bins"), py::arg("n_bins"),
          py::arg("max_depth"), py::arg("reg_lambda"), py::arg("gamma"),
          py::arg("min_child_weight"), py::arg("min_samples_leaf"),
          py::arg("n_threads") = -1,
          "Grow one level-wise histogram tree with histogram subtraction.");
}
