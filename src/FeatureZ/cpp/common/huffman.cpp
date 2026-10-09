#include "common/huffman.hpp"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <queue>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

constexpr std::uint8_t kMagic[4] = {'H', 'F', 'N', '1'};

struct Node {
    std::int32_t symbol = 0;
    std::uint64_t freq = 0;
    std::shared_ptr<Node> left;
    std::shared_ptr<Node> right;
    bool leaf = false;
};

struct NodeCompare {
    bool operator()(const std::shared_ptr<Node>& a, const std::shared_ptr<Node>& b) const {
        if (a->freq != b->freq) {
            return a->freq > b->freq;
        }
        return a->symbol > b->symbol;
    }
};

void append_u32(std::vector<std::uint8_t>& out, std::uint32_t v) {
    out.push_back(static_cast<std::uint8_t>((v >> 0) & 0xFFu));
    out.push_back(static_cast<std::uint8_t>((v >> 8) & 0xFFu));
    out.push_back(static_cast<std::uint8_t>((v >> 16) & 0xFFu));
    out.push_back(static_cast<std::uint8_t>((v >> 24) & 0xFFu));
}

void append_u64(std::vector<std::uint8_t>& out, std::uint64_t v) {
    for (int i = 0; i < 8; ++i) {
        out.push_back(static_cast<std::uint8_t>((v >> (8 * i)) & 0xFFu));
    }
}

void append_i32(std::vector<std::uint8_t>& out, std::int32_t v) {
    append_u32(out, static_cast<std::uint32_t>(v));
}

std::uint32_t read_u32(const std::vector<std::uint8_t>& in, std::size_t& offset) {
    if (offset + 4 > in.size()) {
        throw std::runtime_error("Invalid Huffman stream: truncated u32.");
    }
    const std::uint32_t v = static_cast<std::uint32_t>(in[offset]) |
                            (static_cast<std::uint32_t>(in[offset + 1]) << 8) |
                            (static_cast<std::uint32_t>(in[offset + 2]) << 16) |
                            (static_cast<std::uint32_t>(in[offset + 3]) << 24);
    offset += 4;
    return v;
}

std::uint64_t read_u64(const std::vector<std::uint8_t>& in, std::size_t& offset) {
    if (offset + 8 > in.size()) {
        throw std::runtime_error("Invalid Huffman stream: truncated u64.");
    }
    std::uint64_t v = 0;
    for (int i = 0; i < 8; ++i) {
        v |= static_cast<std::uint64_t>(in[offset + i]) << (8 * i);
    }
    offset += 8;
    return v;
}

std::int32_t read_i32(const std::vector<std::uint8_t>& in, std::size_t& offset) {
    return static_cast<std::int32_t>(read_u32(in, offset));
}

std::shared_ptr<Node> build_tree_from_freq(const std::map<std::int32_t, std::uint64_t>& freq_by_symbol) {
    std::priority_queue<std::shared_ptr<Node>, std::vector<std::shared_ptr<Node>>, NodeCompare> pq;

    for (const auto& entry : freq_by_symbol) {
        auto node = std::make_shared<Node>();
        node->symbol = entry.first;
        node->freq = entry.second;
        node->leaf = true;
        pq.push(node);
    }

    if (pq.empty()) {
        return nullptr;
    }

    while (pq.size() > 1) {
        auto left = pq.top();
        pq.pop();
        auto right = pq.top();
        pq.pop();

        auto parent = std::make_shared<Node>();
        parent->symbol = std::min(left->symbol, right->symbol);
        parent->freq = left->freq + right->freq;
        parent->left = left;
        parent->right = right;
        parent->leaf = false;
        pq.push(parent);
    }

    return pq.top();
}

void build_codes(
    const std::shared_ptr<Node>& node,
    std::vector<bool>& prefix,
    std::unordered_map<std::int32_t, std::vector<bool>>& codes
) {
    if (!node) {
        return;
    }
    if (node->leaf) {
        codes[node->symbol] = prefix;
        return;
    }

    prefix.push_back(false);
    build_codes(node->left, prefix, codes);
    prefix.pop_back();

    prefix.push_back(true);
    build_codes(node->right, prefix, codes);
    prefix.pop_back();
}

} // namespace

std::vector<std::uint8_t> huffman_encode(const std::vector<std::int32_t>& values) {
    std::map<std::int32_t, std::uint64_t> freq_by_symbol;
    for (std::int32_t value : values) {
        ++freq_by_symbol[value];
    }

    std::vector<std::uint8_t> out;
    out.insert(out.end(), std::begin(kMagic), std::end(kMagic));

    if (freq_by_symbol.size() > static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max())) {
        throw std::runtime_error("Too many unique symbols for Huffman serialization.");
    }
    append_u32(out, static_cast<std::uint32_t>(freq_by_symbol.size()));

    for (const auto& entry : freq_by_symbol) {
        append_i32(out, entry.first);
        append_u64(out, entry.second);
    }

    if (values.empty()) {
        append_u64(out, 0);
        return out;
    }

    auto root = build_tree_from_freq(freq_by_symbol);
    if (!root) {
        throw std::runtime_error("Failed to build Huffman tree.");
    }

    if (freq_by_symbol.size() == 1) {
        append_u64(out, 0);
        return out;
    }

    std::unordered_map<std::int32_t, std::vector<bool>> codes;
    std::vector<bool> prefix;
    build_codes(root, prefix, codes);

    std::vector<std::uint8_t> payload;
    payload.reserve(values.size());

    std::uint8_t current = 0;
    int bit_index = 0;
    std::uint64_t bit_count = 0;

    for (std::int32_t value : values) {
        const auto it = codes.find(value);
        if (it == codes.end()) {
            throw std::runtime_error("Internal Huffman error: missing symbol code.");
        }
        for (bool bit : it->second) {
            if (bit) {
                current |= static_cast<std::uint8_t>(1u << (7 - bit_index));
            }
            ++bit_index;
            ++bit_count;
            if (bit_index == 8) {
                payload.push_back(current);
                current = 0;
                bit_index = 0;
            }
        }
    }
    if (bit_index != 0) {
        payload.push_back(current);
    }

    append_u64(out, bit_count);
    out.insert(out.end(), payload.begin(), payload.end());
    return out;
}

std::vector<std::int32_t> huffman_decode(const std::vector<std::uint8_t>& encoded) {
    std::size_t offset = 0;
    if (encoded.size() < 4) {
        throw std::runtime_error("Invalid Huffman stream: too short.");
    }
    for (int i = 0; i < 4; ++i) {
        if (encoded[i] != kMagic[i]) {
            throw std::runtime_error("Invalid Huffman stream: bad magic.");
        }
    }
    offset = 4;

    const std::uint32_t symbol_count = read_u32(encoded, offset);
    std::map<std::int32_t, std::uint64_t> freq_by_symbol;
    std::uint64_t total_values = 0;

    for (std::uint32_t i = 0; i < symbol_count; ++i) {
        const std::int32_t symbol = read_i32(encoded, offset);
        const std::uint64_t freq = read_u64(encoded, offset);
        freq_by_symbol[symbol] = freq;
        total_values += freq;
    }

    const std::uint64_t bit_count = read_u64(encoded, offset);
    const std::size_t payload_bytes_needed = static_cast<std::size_t>((bit_count + 7u) / 8u);
    if (offset + payload_bytes_needed > encoded.size()) {
        throw std::runtime_error("Invalid Huffman stream: payload truncated.");
    }

    std::vector<std::int32_t> out;
    out.reserve(static_cast<std::size_t>(total_values));

    if (total_values == 0) {
        return out;
    }

    auto root = build_tree_from_freq(freq_by_symbol);
    if (!root) {
        throw std::runtime_error("Invalid Huffman stream: missing tree.");
    }

    if (root->leaf) {
        out.assign(static_cast<std::size_t>(total_values), root->symbol);
        return out;
    }

    std::uint64_t consumed_bits = 0;
    std::size_t payload_offset = offset;
    std::shared_ptr<Node> current = root;

    while (consumed_bits < bit_count && out.size() < static_cast<std::size_t>(total_values)) {
        const std::size_t byte_index = static_cast<std::size_t>(consumed_bits / 8u);
        const int bit_in_byte = static_cast<int>(consumed_bits % 8u);
        const std::uint8_t byte = encoded[payload_offset + byte_index];
        const bool bit = ((byte >> (7 - bit_in_byte)) & 0x1u) != 0;
        ++consumed_bits;

        current = bit ? current->right : current->left;
        if (!current) {
            throw std::runtime_error("Invalid Huffman stream: traversal failed.");
        }
        if (current->leaf) {
            out.push_back(current->symbol);
            current = root;
        }
    }

    if (out.size() != static_cast<std::size_t>(total_values)) {
        throw std::runtime_error("Invalid Huffman stream: decoded length mismatch.");
    }
    return out;
}
