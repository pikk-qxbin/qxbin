/*
 * QxBin CUDA Python Bindings (pybind11)
 * Exposes the native high-performance CUDA kernel to Python
 *
 * Build (recommended):
 *   pip install pybind11 numpy
 *   c++ -O3 -shared -std=c++17 -fPIC $(python3 -m pybind11 --includes) \
 *       qxbin_cuda_pybind.cpp -o qxbin_cuda$(python3-config --extension-suffix) \
 *       -L/usr/local/cuda/lib64 -lcudart -lcuda
 *
 * Or use the provided setup.py / pyproject.toml in a real package.
 *
 * This gives you the speed of the native .cu kernel while staying in Python.
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <cuda_runtime.h>
#include <vector>
#include <stdexcept>

namespace py = pybind11;

// Forward declaration of the CUDA kernel (from qxbin_cuda.cu)
extern "C" void evolve_step(
    float* d_states,
    float* d_biases,
    int* d_ns,
    int* d_ms,
    int num_cubits,
    int grid_size
);

// Simple wrapper class that owns GPU memory and exposes Python-friendly API
class QxBinNative {
public:
    QxBinNative(int num_cubits = 256, int grid_size = 8)
        : num_cubits_(num_cubits), grid_size_(grid_size) {

        matrix_size_ = grid_size_ * grid_size_;
        states_bytes_ = num_cubits_ * matrix_size_ * sizeof(float);
        vec_bytes_ = num_cubits_ * sizeof(float);
        int_vec_bytes_ = num_cubits_ * sizeof(int);

        // Allocate device memory
        cudaMalloc(&d_states_, states_bytes_);
        cudaMalloc(&d_biases_, vec_bytes_);
        cudaMalloc(&d_ns_, int_vec_bytes_);
        cudaMalloc(&d_ms_, int_vec_bytes_);

        // Initialize random states on GPU (simple host init + copy for v1)
        std::vector<float> h_states(num_cubits_ * matrix_size_);
        std::vector<float> h_biases(num_cubits_);
        std::vector<int> h_ns(num_cubits_);
        std::vector<int> h_ms(num_cubits_);

        srand(42);
        for (int c = 0; c < num_cubits_; c++) {
            float sum = 0.0f;
            for (int i = 0; i < matrix_size_; i++) {
                float val = (float)rand() / RAND_MAX;
                h_states[c * matrix_size_ + i] = val;
                sum += val;
            }
            for (int i = 0; i < matrix_size_; i++) {
                h_states[c * matrix_size_ + i] /= sum;
            }
            h_biases[c] = 0.5f + ((float)rand() / RAND_MAX) * 0.35f;
            h_ns[c] = 1 + rand() % 5;
            h_ms[c] = 1 + rand() % 5;
        }

        cudaMemcpy(d_states_, h_states.data(), states_bytes_, cudaMemcpyHostToDevice);
        cudaMemcpy(d_biases_, h_biases.data(), vec_bytes_, cudaMemcpyHostToDevice);
        cudaMemcpy(d_ns_, h_ns.data(), int_vec_bytes_, cudaMemcpyHostToDevice);
        cudaMemcpy(d_ms_, h_ms.data(), int_vec_bytes_, cudaMemcpyHostToDevice);
    }

    ~QxBinNative() {
        cudaFree(d_states_);
        cudaFree(d_biases_);
        cudaFree(d_ns_);
        cudaFree(d_ms_);
    }

    // Evolve all cubit chains on GPU (calls the native kernel)
    py::array_t<float> evolve_chains() {
        // Call the CUDA kernel (defined in qxbin_cuda.cu)
        evolve_step(d_states_, d_biases_, d_ns_, d_ms_, num_cubits_, grid_size_);

        // Return aggregate mean as numpy array (for compatibility with viz & other tiers)
        std::vector<float> h_agg(matrix_size_, 0.0f);
        // For simplicity in v1 we compute mean on host after copy
        // (in production we'd keep more on GPU)
        std::vector<float> h_states(num_cubits_ * matrix_size_);
        cudaMemcpy(h_states.data(), d_states_, states_bytes_, cudaMemcpyDeviceToHost);

        for (int c = 0; c < num_cubits_; c++) {
            for (int i = 0; i < matrix_size_; i++) {
                h_agg[i] += h_states[c * matrix_size_ + i];
            }
        }
        for (int i = 0; i < matrix_size_; i++) {
            h_agg[i] /= num_cubits_;
        }

        return py::array_t<float>({grid_size_, grid_size_}, h_agg.data());
    }

    // Full optimization loop (GPU accelerated)
    py::array_t<float> optimize_to_target(float target_mean = 0.7f, int max_steps = 200) {
        for (int step = 0; step < max_steps; step++) {
            auto agg = evolve_chains();
            float current = 0.0f;
            auto buf = agg.request();
            float* ptr = static_cast<float*>(buf.ptr);
            for (int i = 0; i < matrix_size_; i++) current += ptr[i];
            current /= matrix_size_;

            if (std::abs(current - target_mean) < 0.001f) {
                py::print("Converged after", step, "steps | mean prob ≈", current);
                break;
            }
        }
        return evolve_chains(); // return final aggregate
    }

    int num_cubits() const { return num_cubits_; }
    int grid_size() const { return grid_size_; }

private:
    int num_cubits_;
    int grid_size_;
    int matrix_size_;
    size_t states_bytes_;
    size_t vec_bytes_;
    size_t int_vec_bytes_;

    float* d_states_ = nullptr;
    float* d_biases_ = nullptr;
    int*   d_ns_ = nullptr;
    int*   d_ms_ = nullptr;
};

PYBIND11_MODULE(qxbin_cuda, m) {
    m.doc() = "QxBin Native CUDA Python Bindings - Maximum performance QxBin on GPU";

    py::class_<QxBinNative>(m, "QxBinNative")
        .def(py::init<int, int>(), py::arg("num_cubits") = 256, py::arg("grid_size") = 8)
        .def("evolve_chains", &QxBinNative::evolve_chains,
             "Evolve all cubit probability matrices on GPU using native CUDA kernel")
        .def("optimize_to_target", &QxBinNative::optimize_to_target,
             py::arg("target_mean") = 0.7f, py::arg("max_steps") = 200,
             "Run GPU-accelerated optimization loop toward target mean probability")
        .def_property_readonly("num_cubits", &QxBinNative::num_cubits)
        .def_property_readonly("grid_size", &QxBinNative::grid_size)
        .def("__repr__", [](const QxBinNative& self) {
            return "<QxBinNative num_cubits=" + std::to_string(self.num_cubits()) +
                   " grid_size=" + std::to_string(self.grid_size()) + ">";
        });
}