#pragma once

#include <vector>
#include <sys/types.h>
#include <cmath>

constexpr int32_t MAX_PRECISION = 8;

// structs

template<typename T>
struct ublb{
    T ub;
    T lb;
};

template<typename T>
struct point_quantization{
    bool success;
    int32_t quantization_number;
    T value;
};

template<typename T>
struct quantization_constants_for_precision{
    T inverse_interval_width;
    int32_t quantization_stride;
    T un_quantize_constant;
    T local_unquantize_constant;
    T adjusted_bound;
};

template<typename T>
using quantization_constants = std::array<quantization_constants_for_precision<T>,MAX_PRECISION+1>;

// inline helpers

template<typename T>
inline T get_inverse_interval_width(T xi, int32_t precision){
    return std::ldexp(1 / xi, precision-1);
}

inline int32_t get_quantization_stride(int32_t precision){
    return 1 << (MAX_PRECISION - precision);
}

template<typename T>
inline T get_un_quantize_constant(T xi){
    return std::ldexp(2*xi,-MAX_PRECISION);
}

template<typename T>
inline T get_local_un_quantize_constant(T xi, int32_t precision){
    return std::ldexp(2*xi, -precision);
}

template<typename T>
inline T get_adjusted_bound(T xi, int32_t precision){
    return std::ldexp(xi, -precision);
}

template<typename T>
inline quantization_constants_for_precision<T> get_quantization_constants_for_precision(
    T xi,
    int32_t precision
){
    return { get_inverse_interval_width(xi, precision), get_quantization_stride(precision), get_un_quantize_constant(xi),
             get_local_un_quantize_constant(xi, precision), get_adjusted_bound(xi, precision) };
}

template<typename T>
inline quantization_constants<T> get_quantization_constants(
    T xi
){
    std::array<quantization_constants_for_precision<T>, MAX_PRECISION+1> constants;
    for( int i = 0; i <= MAX_PRECISION; ++i ){
        constants[i] = get_quantization_constants_for_precision(xi, i);
    }

    return constants;
}

template<typename T>
inline T un_quantize(T de_val, int32_t quantization_number, T un_quantize_constant){
    if( quantization_number == 0 ){
        return de_val;
    } else {
        return de_val + quantization_number * un_quantize_constant;
    }
}

// types of quantization

template<typename T, bool check_correctness>
inline point_quantization<T> quantize_point_most_accurate(
    T gt,
    T de,
    const ublb<T>& bounds,
    const quantization_constants_for_precision<T>& constants
){
    point_quantization<T> q;
    q.success = true;

    T offset = (gt - de) * constants.inverse_interval_width;
    int32_t local_quantization_number = std::round(offset);
    q.quantization_number = constants.quantization_stride * local_quantization_number;
    q.value = un_quantize(de, q.quantization_number, constants.un_quantize_constant);

    if constexpr( check_correctness ){

        if( q.value <= bounds.lb ){

            q.quantization_number += constants.quantization_stride;
            q.value = un_quantize(de, q.quantization_number, constants.un_quantize_constant);

            if( q.value >= bounds.ub ){
                q.success = false;
            }

        } else if( q.value >= bounds.ub ){

            q.quantization_number -= constants.quantization_stride;
            q.value = un_quantize(de, q.quantization_number, constants.un_quantize_constant);

            if( q.value <= bounds.lb ){
                q.success = false;
            }

        }

    }

    return q;

}

template<typename T, bool check_correctness>
inline point_quantization<T> quantize_point_lowest_number(
    T gt,
    T de,
    const ublb<T>& bounds,
    const quantization_constants_for_precision<T>& constants
){

    point_quantization<T> q;
    q.success = true;

    if( de >= bounds.ub ){

        T offset = (bounds.ub - de) * constants.inverse_interval_width;
        int32_t local_quantization_number = std::floor(offset);
        q.quantization_number = local_quantization_number * constants.quantization_stride;
        q.value = un_quantize(de, q.quantization_number, constants.un_quantize_constant);

        if constexpr( check_correctness ){
            if( q.value >= bounds.ub || q.value <= bounds.lb ){
                q.success = false;
            }
        }
    } else if( de <= bounds.lb ){

        T offset = (bounds.lb - de) * constants.inverse_interval_width;
        int32_t local_quantization_number = std::ceil(offset);
        q.quantization_number = local_quantization_number * constants.quantization_stride;
        q.value = un_quantize(de, q.quantization_number, constants.un_quantize_constant);

        if constexpr( check_correctness ){
            if( q.value >= bounds.ub || q.value <= bounds.lb ){
                q.success = false;
            }
        }
    } else {
        q.quantization_number = 0;
        q.value = de;
    }

    return q;
}

template<typename T, bool most_accurate>
inline point_quantization<T> quantize_for_precision_or_lossless(
    T gt,
    T de,
    const ublb<T>& bounds,
    int32_t bounds_precision,
    int32_t scale_precision,
    const quantization_constants<T>& all_constants
){

    const quantization_constants_for_precision<T>& bounds_constants = all_constants[bounds_precision];
    const quantization_constants_for_precision<T>& scale_constants = all_constants[scale_precision];

    ublb<T> tighter_bounds = { std::min(bounds.ub, gt + bounds_constants.adjusted_bound),
                        std::max(bounds.lb, gt - bounds_constants.adjusted_bound) };

    point_quantization<T> pq;

    if constexpr( most_accurate ){
        pq = quantize_point_most_accurate<T, true>(gt, de, tighter_bounds, scale_constants);
    } else {
        pq = quantize_point_lowest_number<T, true>(gt, de, tighter_bounds, scale_constants);
    }

    if( !pq.success ){
        pq.quantization_number = LOSSLESS_SYMBOL;
        pq.value = gt;
        pq.success = true;
    }

    return pq;

}

template<typename T, bool most_accurate, bool check_correctness>
point_quantization<T> quantize_point(
    T gt,
    T de,
    const ublb<T>& bounds,
    const quantization_constants_for_precision<T>& constants
){
    if constexpr(most_accurate){
        return quantize_point_most_accurate<T, check_correctness>(gt, de, bounds, constants);
    } else {
        return quantize_point_lowest_number<T, check_correctness>(gt, de, bounds, constants);
    }
}
