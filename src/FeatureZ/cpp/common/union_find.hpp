#include <unordered_map>
#include <vector>
#include <stdexcept>
#include <functional>

// chatGPT
template <typename Key, typename Hash = std::hash<Key>, typename KeyEqual = std::equal_to<Key>>
class UnionFind {
private:
    std::unordered_map<Key, int, Hash, KeyEqual> index_of;
    std::vector<int> parent;
    std::vector<Key> keys;

    int find_index(int x) {
        int root = x;
        while (parent[root] != root) {
            root = parent[root];
        }

        while (parent[x] != x) {
            int next = parent[x];
            parent[x] = root;
            x = next;
        }

        return root;
    }

public:
    bool contains(const Key& key) const {
        return index_of.find(key) != index_of.end();
    }

    int add(const Key& key) {
        auto it = index_of.find(key);
        if (it != index_of.end()) {
            return it->second;
        }

        int id = static_cast<int>(parent.size());
        index_of.emplace(key, id);
        parent.push_back(id);
        keys.push_back(key);
        return id;
    }

    const Key& find(const Key& key) {
        auto it = index_of.find(key);
        if (it == index_of.end()) {
            throw std::out_of_range("UnionFind::find: key not found");
        }

        int root = find_index(it->second);
        return keys[root];
    }

    bool connected(const Key& a, const Key& b) {
        auto ita = index_of.find(a);
        auto itb = index_of.find(b);

        if (ita == index_of.end() || itb == index_of.end()) {
            return false;
        }

        return find_index(ita->second) == find_index(itb->second);
    }

    // Make b's root the parent/root of a's root.
    void unite_under(const Key& a, const Key& b) {
        int ia = add(a);
        int ib = add(b);

        int ra = find_index(ia);
        int rb = find_index(ib);

        if (ra == rb) {
            return;
        }

        parent[ra] = rb;
    }
};