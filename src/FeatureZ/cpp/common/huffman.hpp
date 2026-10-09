#pragma once

#include <cstdint>
#include <vector>

std::vector<std::uint8_t> huffman_encode(const std::vector<std::int32_t>& values);
std::vector<std::int32_t> huffman_decode(const std::vector<std::uint8_t>& encoded);
