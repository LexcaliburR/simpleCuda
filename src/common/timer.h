/*
 * @Author: lexcaliburr li289427380@gmail.com
 * @Date: 2023-07-08 18:44:21
 * @LastEditors: lexcaliburr li289427380@gmail.com
 * @LastEditTime: 2023-07-09 15:59:43
 * @FilePath: /simpleCuda/src/common/timer.h
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置
 * 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
 */
#pragma once

#include <vector>
#include <string>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

namespace simplecuda {

class LatencyTimer
{
public:
    LatencyTimer(const std::string& mark, bool use_cuda = false,
                 bool with_raw = false);
    void Start();
    void Toc();
#ifdef USE_CUDA
    void Start(cudaStream_t stream);
    void Toc(cudaStream_t stream);
#endif
    double Max() const;
    double Mean() const;
    double Min() const;
    double STD() const;
    double SUM() const;
    void Print() const;
    void SaveToFile() const;

private:
    double GetCurrentTime() const;

private:
    std::vector<double> delays_;
    cudaEvent_t start_;
    cudaEvent_t stop_;
    bool use_cuda_ = false;
    bool with_raw_ = false;
    double pre_time_ = 0.0;
    std::string mark_ = "";
};

}  // namespace simplecuda