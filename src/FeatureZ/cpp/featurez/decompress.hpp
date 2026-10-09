#include "featurez/include.hpp"

template<typename T>
struct compression_output{
    std::vector<int32_t> quantization_numbers;
    std::vector<T> lossless_storage;
};

template<typename T>
compression_output<T> bytes_to_compression_output(
    const py::bytes& compressed_bytes
){
    std::string in = compressed_bytes;

    if( in.size() < static_cast<ssize_t>(sizeof(std::uint64_t)) ){
        throw std::runtime_error("Invalid compression output: missing Huffman size header.");
    }

    std::size_t off = 0;
    std::uint64_t num_huffman_bytes = 0;
    std::memcpy(&num_huffman_bytes, in.data() + off, sizeof(num_huffman_bytes));
    off += sizeof(num_huffman_bytes);

    if( num_huffman_bytes > static_cast<std::uint64_t>(in.size() - off) ){
        throw std::runtime_error("Invalid compression output: Huffman payload exceeds input size.");
    }

    std::vector<std::uint8_t> huffman_bytes(static_cast<std::size_t>(num_huffman_bytes));
    if( num_huffman_bytes > 0 ){
        std::memcpy(huffman_bytes.data(), in.data() + off, static_cast<std::size_t>(num_huffman_bytes));
        off += static_cast<std::size_t>(num_huffman_bytes);
    }

    compression_output<T> output;
    output.quantization_numbers = huffman_decode(huffman_bytes);

    const std::size_t remaining = static_cast<std::size_t>(in.size()) - off;
    if( remaining % sizeof(T) != 0 ){
        throw std::runtime_error("Invalid compression output: lossless payload size is not aligned to element size.");
    }

    const std::size_t num_lossless = remaining / sizeof(T);
    output.lossless_storage.resize(num_lossless);
    if( num_lossless > 0 ){
        std::memcpy(output.lossless_storage.data(), in.data() + off, remaining);
    }

    return output;
}

template<typename T>
py::array_t<T> featurez_decompress_impl_template(
    const py::array_t<T, py::array::f_style | py::array::forcecast>& de_sf_py,
    py::bytes compressed_bytes,
    T xi
){

    compression_output<T> output = bytes_to_compression_output<T>(compressed_bytes);

    ssize_t ndim = de_sf_py.ndim();
    if( ndim != 2 && ndim != 3 ){
        throw std::runtime_error("Feature preserving decompressor only supports 2D or 3D scalar fields.");
    }

    ssize_t size_x = de_sf_py.shape(0);
    ssize_t size_y = de_sf_py.shape(1);
    ssize_t size_z = (ndim == 3) ? de_sf_py.shape(2) : 1;
    ssize_t num_points = size_x * size_y * size_z;

    py::buffer_info de_info = de_sf_py.request();
    const T* de_sf = static_cast<T*>(de_info.ptr);

    py::array_t<T, py::array::f_style> re_sf_py =
        (ndim == 3) ? py::array_t<T, py::array::f_style>({size_x, size_y, size_z})
                    : py::array_t<T, py::array::f_style>({size_x, size_y});
    T* re_sf = re_sf_py.mutable_data();

    ssize_t next_lossless = 0;

    T un_quantize_constant = get_un_quantize_constant(xi);

    for( ssize_t i = 0; i < num_points; ++i ){

        int32_t quantization_number = output.quantization_numbers[i];

        if( quantization_number == LOSSLESS_SYMBOL ){
            re_sf[i] = output.lossless_storage[next_lossless];
            ++next_lossless;
        } else {
            re_sf[i] = un_quantize(de_sf[i], quantization_number, un_quantize_constant );
        }

    }

    return re_sf_py;

}