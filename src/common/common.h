/*
 * @Author: lexcaliburr li289427380@gmail.com
 * @Date: 2023-07-08 18:45:21
 * @LastEditors: lexcaliburr li289427380@gmail.com
 * @LastEditTime: 2023-07-09 16:18:58
 * @FilePath: /simpleCuda/src/common/common.h
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置
 * 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
 */
#pragma once

#include <cuda_runtime.h>
#include <iostream>
#include <chrono>

// #include "common/check_utils.h"
#include "common/timer.h"

#define DIVUP(x, y) (x - 1) / y + 1
#define GET_CURR_TIME()                                               \
    std::chrono::duration_cast<std::chrono::microseconds>(            \
        std::chrono::high_resolution_clock::now().time_since_epoch()) \
            .count() *                                                \
        1e-3

namespace simplecuda {

template <class T>
void PrintArray(T* f_data, size_t f_totalNum)
{
    for (int i = 0; i < f_totalNum; i++) {
        std::cout << f_data[i] << " ";
    }
    std::cout << std::endl;
}

class _TicToc
{
public:
    _TicToc(const std::string& f_mark, const std::string& f_funcMark)
    {
        m_mark = f_funcMark + " " + f_mark;
        m_toc = GET_CURR_TIME();
    }
    void Toc()
    {
        //
        double cur_time = GET_CURR_TIME();

        std::cout << m_mark << ": " << cur_time - m_toc << "ms" << std::endl;
        m_toc = cur_time;
    }

private:
    double m_toc{0};
    std::string m_mark{};
};

}  // namespace simplecuda

#define PERF_START(mark) simplecuda::_TicToc _ticToc(mark, __FUNCTION__);
#define PERF_END _ticToc.Toc();