/*
 * @Author: lexcaliburr li289427380@gmail.com
 * @Date: 2023-07-08 18:45:07
 * @LastEditors: lexcaliburr li289427380@gmail.com
 * @LastEditTime: 2023-07-09 16:19:57
 * @FilePath: /simpleCuda/src/common/timer.cpp
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置
 * 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
 */
#include "common/timer.h"
#include "common/common.h"

#include <chrono>
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <thread>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

namespace simplecuda {

LatencyTimer::LatencyTimer(const std::string& mark, bool use_cuda,
                           bool with_raw)
    : mark_(mark), use_cuda_(use_cuda), with_raw_(with_raw)
{
#ifdef USE_CUDA
    if (use_cuda_) {
        CUDA_CHECK(cudaEventCreate(&start_));
        CUDA_CHECK(cudaEventCreate(&stop_));
    }
#endif
}

void LatencyTimer::Start() { pre_time_ = GetCurrentTime(); }

void LatencyTimer::Toc()
{
    double cur_time = GetCurrentTime();
    double delay = cur_time - pre_time_;

    if (use_cuda_) {
#ifdef USE_CUDA
        cudaDeviceSynchronize();
#endif
    }

    pre_time_ = cur_time;
    delays_.push_back(delay);
}

#ifdef USE_CUDA
void LatencyTimer::Start(cudaStream_t stream) {
    cudaEventRecord(start_, stream);
}
void LatencyTimer::Toc(cudaStream_t stream)
{
    cudaEventRecord(stop_, stream);
    cudaEventSynchronize(stop_);
    float delay;
    cudaEventElapsedTime(&delay, start_, stop_);
    delays_.push_back(delay);
}

#endif

double LatencyTimer::Max() const
{
    if (delays_.empty()) return 0.0;
    return *std::max_element(delays_.begin(), delays_.end());
}

double LatencyTimer::Mean() const
{
    if (delays_.empty()) return 0.0;
    double sum = std::accumulate(delays_.begin(), delays_.end(), 0.0);
    return sum / delays_.size();
}

double LatencyTimer::Min() const
{
    if (delays_.empty()) return 0.0;
    return *std::min_element(delays_.begin(), delays_.end());
}

double LatencyTimer::STD() const
{
    if (delays_.size() < 2) return 0.0;
    double mean = Mean();
    double variance = 0.0;
    for (double delay : delays_) {
        double diff = delay - mean;
        variance += diff * diff;
    }
    variance /= delays_.size();
    return std::sqrt(variance);
}

double LatencyTimer::SUM() const
{
    if (delays_.empty()) return 0.0;
    double sum = std::accumulate(delays_.begin(), delays_.end(), 0.0);
    return sum;
}

void LatencyTimer::Print() const
{
    std::cout << "Mark: " << mark_ << std::endl;
    std::cout << "Min Delay: " << Min() << " ms" << std::endl;
    std::cout << "Max Delay: " << Max() << " ms" << std::endl;
    std::cout << "Mean Delay: " << Mean() << " ms" << std::endl;
    std::cout << "Standard Deviation: " << STD() << " ms" << std::endl;
    std::cout << "SUM Delay: " << SUM() << " ms" << std::endl;
}

void LatencyTimer::SaveToFile() const
{
    std::ofstream file("latency_results.txt");
    if (file.is_open()) {
        file << "Mark: " << mark_ << std::endl;
        file << "Min Delay: " << Min() << " ms" << std::endl;
        file << "Max Delay: " << Max() << " ms" << std::endl;
        file << "Mean Delay: " << Mean() << " ms" << std::endl;
        file << "Standard Deviation: " << STD() << " ms" << std::endl;
        file << "SUM Delay: " << SUM() << " ms" << std::endl;


        if (with_raw_) {
            file << "Raw Data:" << std::endl;
            for (double delay : delays_) {
                file << delay << std::endl;
            }
        }

        file.close();
    } else {
        std::cerr << "Error: Failed to open the file." << std::endl;
    }
}

double LatencyTimer::GetCurrentTime() const
{
    return std::chrono::duration_cast<std::chrono::microseconds>(
               std::chrono::high_resolution_clock::now().time_since_epoch())
               .count() *
           1e-3;
}

}  // namespace simplecuda

// int main()
// {
//     simplecuda::LatencyTimer timer("test", false, false);
//     timer.Start();
//     for (int i = 0; i < 100; ++i) {
//         timer.Toc();
//         std::this_thread::sleep_for(std::chrono::milliseconds(10));
//     }
//     timer.Print();
//     timer.SaveToFile();
//     return 0;
// }