/**
 * @file perf.cpp
 * @brief Out-of-line parts of perf_t and cuperf_t (see perf.hpp for the usage pattern)
 * @author GEMM Project
 * @date 2025
 * 
 * Contains the constructors / destructors (instance registration, CUDA event lifetime),
 * cuperf_t::end_timer(), the NVML energy measurement and the static member definitions.
 */

#include <perf.hpp>
#include <fstream>

/// Register the new CPU timer so that perf_t::dump() can find it.
perf_t::perf_t(const std::string &message)
    : message_(message)
{
    hinstances_.insert(this);
}

/**
 * @brief Destructor for CPU performance measurement
 * 
 * Unregisters the instance from the global tracker when destroyed.
 */
perf_t::~perf_t()
{
    auto it = hinstances_.find(this);
    hinstances_.erase(it);
}

/// Register the GPU timer and create its two events (default flags, timing enabled).
cuperf_t::cuperf_t(const std::string &message)
    : message_(message)
{
    cinstances_.insert(this);
    CHECK(cudaEventCreate(&start_));
    CHECK(cudaEventCreate(&stop_));
}

/**
 * @brief Destructor for CUDA performance measurement
 * 
 * Unregisters the instance and destroys CUDA events when destroyed.
 */
cuperf_t::~cuperf_t()
{
    auto it = cinstances_.find(this);
    cinstances_.erase(it);
    // destructor: never throw
    CUDA_WARN(cudaEventDestroy(start_));
    CUDA_WARN(cudaEventDestroy(stop_));
}

/// Record the stop event on @p stream, block until the GPU reaches it, and add the elapsed time
/// since the start event (resolution about 0.5 us per CUDA documentation) as one sample.
float cuperf_t::end_timer(cudaStream_t stream)
{
    CHECK(cudaEventRecord(stop_, stream));
    CHECK(cudaEventSynchronize(stop_));
    float ms = 0.0f;
    CHECK(cudaEventElapsedTime(&ms, start_, stop_));

    minTime_ = std::min(minTime_, ms);
    maxTime_ = std::max(maxTime_, ms);
    totalTime_ += ms;
    ++count_;
    return ms;
}

/**
 * @brief Check an NVML call and log a failure to std::cerr
 * @param call NVML function call to check
 * 
 * Unlike CUDA_CHECK this does not throw: energy measurement is optional, so an NVML error
 * (e.g. counter not supported on this GPU) is only reported and execution continues.
 */
#define NVML_CHECK(call) \
do { \
        const auto err = call; \
        if (nvmlReturn_t::NVML_SUCCESS != err) { \
            std::cerr << "Failed to call NVML API, " << __FILE__ << "," << __LINE__ << ", error=" << err << ", " << nvmlErrorString(err) << std::endl; \
        } \
} while (0)

/// Initialize NVML and take the handle of device 0 (NVML index 0 is assumed to be the same GPU
/// as CUDA device 0). nvmlShutdown() is never called; the library stays loaded until exit.
void cuperf_t::init()
{
    NVML_CHECK(nvmlInit());
    NVML_CHECK(nvmlDeviceGetHandleByIndex(0, &device_));

    char name[64] = {}; // NVML_DEVICE_NAME_BUFFER_SIZE (64)
    NVML_CHECK(nvmlDeviceGetName(device_, name, sizeof(name)));
    std::cerr << "Initialized NVML for device 0: " << name << std::endl;
}

/// Start an energy sample: read the total energy (mJ since the driver was loaded) of the GPU.
/// The read happens now on the host and is not ordered with @p stream.
void cuperf_t::start_monitor_energy(cudaStream_t stream)
{
    // init before any call on dev 0
    if (nullptr == device_) {
        init();
    }

    NVML_CHECK(nvmlDeviceGetTotalEnergyConsumption(device_, &energyBefore_));
    (void)stream;
}

/**
 * @brief Extract the value of an nvmlFieldValue_t (from nvmlDeviceGetFieldValues) as float
 * @param fields NVML field value structure containing the data
 * @return Extracted value as float; asserts on an unknown value type
 * 
 * Converts according to fields.valueType. Currently not called anywhere; kept for querying
 * NVML fields (e.g. power or energy fields) instead of the dedicated energy call.
 */
auto value(const nvmlFieldValue_t& fields) {
    const auto& value = fields.value;
    auto diff = 0.0f;
    switch (fields.valueType) {
    case NVML_VALUE_TYPE_DOUBLE: {
        diff = (float)value.dVal;
    } break;
    case NVML_VALUE_TYPE_UNSIGNED_INT: {
        diff = (float)value.uiVal;
    } break;
    case NVML_VALUE_TYPE_UNSIGNED_LONG: {
        diff = (float)value.ulVal;
    } break;
    case NVML_VALUE_TYPE_UNSIGNED_LONG_LONG: {
        diff = (float)value.ullVal;
    } break;
    case NVML_VALUE_TYPE_SIGNED_LONG_LONG: {
        diff = (float)value.sllVal;
    } break;
    case NVML_VALUE_TYPE_SIGNED_INT: {
        diff = (float)value.siVal;
    } break;
    case NVML_VALUE_TYPE_UNSIGNED_SHORT: {
        diff = (float)value.usVal;
    } break;
    default:
        std::cerr << "Invalid value type: " << fields.valueType << std::endl;
        CV_Assert(false);
    }
    return diff;
}

/// End an energy sample: read the counter again and add the difference (mJ) as one sample.
/// The counter is not updated continuously, so a very short interval may read 0; the sample is
/// clamped to at least 1 mJ.
float cuperf_t::end_monitor_energy(cudaStream_t stream)
{
    (void)stream;
    unsigned long long energyAfter = 0;
    NVML_CHECK(nvmlDeviceGetTotalEnergyConsumption(device_, &energyAfter));
    const auto diff = energyAfter - energyBefore_;
    const auto mJ = std::max(1.0f, float(diff));

    minTime_ = std::min(minTime_, mJ);
    maxTime_ = std::max(maxTime_, mJ);
    totalTime_ += mJ;
    ++count_;
    return mJ;
}

// static members: instance registries and the lazily initialized NVML device handle
std::set<perf_t*> perf_t::hinstances_;
std::set<cuperf_t*> cuperf_t::cinstances_;
nvmlDevice_t cuperf_t::device_ = nullptr;
