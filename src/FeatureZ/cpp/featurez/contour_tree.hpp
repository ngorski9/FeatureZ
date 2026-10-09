#pragma once

#include <memory>
#include <unordered_map>
#include <unordered_set>

#include "include.hpp"
#include "sf/sf_3d.hpp"
#include "common/union_find.hpp"

template<typename T>
struct sf_more {
    const T* sf;

    bool operator()(int32_t i, int32_t j) const {
        return sf[i] > sf[j] || (sf[i] == sf[j] && i > j);
    }

    sf_more(T* sf_) : sf(sf_) {}
};

template<typename T>
struct sf_less {
    const T* sf;

    bool operator()(int32_t i, int32_t j) const {
        return sf[i] < sf[j] || (sf[i] == sf[j] && i < j);
    }

    sf_less(T* sf_) : sf(sf_) {}
};

template<typename T>
struct Node{
    int32_t index;
    T value;
    std::vector<int32_t> segments;
    std::shared_ptr<Node<T>> parent;
    std::shared_ptr<Node<T>> child_towards_extremum;
    std::vector<std::shared_ptr<Node<T>>> children;
    int8_t num_paired = 0;

    Node(int32_t index_, T value_, int32_t segment) : index(index_), value(value_) {
        segments.push_back(segment);
    }
};

template<typename T>
using node_ptr = std::shared_ptr<Node<T>>;

template<typename T>
using tree = std::unordered_map<int32_t, node_ptr<T>>;

template<typename T, bool join>
bool compare_nodes( node_ptr<T> try_less, node_ptr<T> try_more ){

    if constexpr (join){
        return (try_more->value > try_less->value) || (try_more->value == try_less->value && try_more->index > try_less->index);
    } else {
        return (try_more->value < try_less->value) || (try_more->value == try_less->value && try_more->index < try_less->index);
    }

}

template<typename T>
tree<T> extract_tree(int32_t* points, T* sf, size_t num_nodes){

    tree<T> t;

    for( size_t i = 0; i < num_nodes; ++i ){
        int32_t index = points[i];
        T value = sf[index];

        t[index] = std::make_shared<Node<T>>(index, value, index);

    }

    for( size_t i = 0; i < num_nodes; ++i ){
        int32_t index = points[i];
        int32_t parent_index = points[i + num_nodes];

        if( parent_index != -1 ){
            node_ptr<T> node = t[index];
            node_ptr<T> parent_node = t[parent_index];

            node->parent = parent_node;
            parent_node->children.push_back(node);
        }
    }

    return t;

}

template<typename T, bool join>
void process_leaf(tree<T>& t, UnionFind<int32_t>& segmentation_conversion, int32_t leaf_index, T epsilon){
    node_ptr<T> leaf_node = t[leaf_index];
    node_ptr<T> parent_saddle = leaf_node->parent;
    node_ptr<T> parent_saddle_child = leaf_node; // used for child path towards extremum

    parent_saddle->child_towards_extremum = parent_saddle_child;

    bool skip_pruning = false;

    // find persistence pair
    while( parent_saddle->num_paired == parent_saddle->children.size() - 1 && parent_saddle->children.size() > 1){
        parent_saddle_child = parent_saddle;
        parent_saddle = parent_saddle->parent;
        parent_saddle->child_towards_extremum = parent_saddle_child;
        skip_pruning = true; // if this happens then we know the persistence is above epsilon       

    }

    // compute persistence
    T persistence;

    if( !skip_pruning ){
        if constexpr ( join ){
            persistence = parent_saddle->value - leaf_node->value;
        } else {
            persistence = leaf_node->value - parent_saddle->value;
        }
    }

    if( !skip_pruning && persistence < epsilon ){
        //prune the pair

        t.erase(leaf_index);

        // avoid a memory leak
        leaf_node->parent = nullptr;
        std::vector<node_ptr<T>>& pc = parent_saddle->children; // parent children
        pc.erase(std::remove(pc.begin(), pc.end(), leaf_node), pc.end());

        if( parent_saddle->children.size() >= 2){
            // monkey saddle
            segmentation_conversion.unite_under(leaf_index, parent_saddle->index);

        } else {
            // normal saddle

            int32_t parent_index = parent_saddle->index;
            node_ptr<T> grandparent = parent_saddle->parent;
            node_ptr<T> sibling = (parent_saddle->children)[0];

            parent_saddle->children.clear();
            parent_saddle->parent = nullptr;
            t.erase(parent_saddle->index);

            sibling->parent = grandparent;

            std::vector<node_ptr<T>>& gpc = grandparent->children;
            gpc.erase(std::remove(gpc.begin(), gpc.end(), parent_saddle), gpc.end());
            gpc.push_back(sibling);



            segmentation_conversion.unite_under(leaf_index, sibling->index);
            segmentation_conversion.unite_under(parent_index, sibling->index);

        }

    } else {
        // keep the pair
        parent_saddle->num_paired += 1;
    }
}

template<typename T, bool join>
UnionFind<int32_t> persistence_pairing_and_simplification(tree<T>& t, T* sf, int32_t* segmentation, size_t num_points, T epsilon){

    UnionFind<int32_t> segmentation_conversion;

    std::vector<int32_t> leaf_indices;

    for( const auto& kv : t ){

        int32_t node_index = kv.first;
        node_ptr<T> node = kv.second;

        if( node->children.size() == 0 ){
            leaf_indices.push_back(node_index);
        }

        segmentation_conversion.add(node_index);
    }

    if constexpr ( join ){
        std::sort( leaf_indices.begin(), leaf_indices.end(), sf_more<T>(sf) );
    } else {
        std::sort( leaf_indices.begin(), leaf_indices.end(), sf_less<T>(sf) );
    }

    for( const int32_t leaf_index : leaf_indices ){
        process_leaf<T,join>(t, segmentation_conversion, leaf_index, epsilon);
    }

    return segmentation_conversion;

}

template<typename T>
tree<T> clone_tree(tree<T>& t){

    tree<T> clone;

    // add nodes
    for( const auto& kv : t ){
        int32_t index = kv.first;
        node_ptr<T> node = kv.second;
        clone[index] = std::make_shared<Node<T>>(index, node->value, (node->segments)[0]);
    }

    // attach parents and children
    for( const auto& kv : t ){
        int32_t index = kv.first;
        node_ptr<T> node = kv.second;

        if( node->parent != nullptr ){
            int32_t parent_index = node->parent->index;

            node_ptr<T> clone_node = clone[index];
            node_ptr<T> clone_parent = clone[parent_index];
            clone_node->parent = clone_parent;
            clone_parent->children.push_back(clone_node);
        }
    }

    return clone;

}

struct PairHash {
    template <class T1, class T2>
    std::size_t operator()(const std::pair<T1, T2>& p) const {
        std::size_t h1 = std::hash<T1>{}(p.first);
        std::size_t h2 = std::hash<T2>{}(p.second);

        return h1 ^ (h2 << 1);
    }
};

template<typename T>
using bounds_map = std::unordered_map< std::pair<int32_t,int32_t>, std::pair<T,T>, PairHash >;

template<typename T>
struct CombineTreesReturn{
    bounds_map<T> bounds;
    tree<T> ct;
};


// btw you need the union find for merging here!
template<typename T, bool join>
void augment_merge_tree(tree<T>& to, tree<T>& from, int32_t* segmentation, UnionFind<int32_t>& segmentation_merge){

    for( const auto& kv : from ){
        int32_t index = kv.first;

        // check that the node isn't already shared by both trees
        if( !to.count(index) ){

            node_ptr<T> node = kv.second;
            int32_t segment = segmentation_merge.find(segmentation[index]);
            
            node_ptr<T> new_node = std::make_shared<Node<T>>(index, node->value, segment);
            node_ptr<T> next_child = to[segment];

            // walk up the segment until we find two adjacent nodes that straddle the node that we are inserting.
            while( !compare_nodes<T, join>(new_node, next_child->parent) ){
                next_child = next_child->parent;
            }

            node_ptr<T> next_parent = next_child->parent;

            std::vector<node_ptr<T>>& pc = next_parent->children; // parent children
            pc.erase(std::remove(pc.begin(), pc.end(), next_child), pc.end());
            pc.push_back(new_node);

            new_node->parent = next_parent;
            new_node->children.push_back(next_child);

            next_child->parent = new_node;
            to[index] = new_node;

        }

    }

}

template<typename T, bool join>
void add_ct_segmentation(
    node_ptr<T> leaf_node,
    tree<T>& interior_tree,
    bounds_map<T>& bounds
){

    std::pair<T,T> bounds_pair;

    if constexpr (join){
        bounds_pair.first = leaf_node->parent->value;
        bounds_pair.second = leaf_node->value;
    } else {
        bounds_pair.first = leaf_node->value;
        bounds_pair.second = leaf_node->parent->value;
    }

    // if( bounds_pair.first == bounds_pair.second ){
    //     std::cout << "zero width bounds" << std::endl;
    //     std::cout << leaf_node->index << " " << leaf_node->value << std::endl;
    //     std::cout << leaf_node->parent->index << " " << leaf_node->parent->value << std::endl;
    // }

    std::vector<int32_t> leaf_segments = leaf_node->segments;
    std::unordered_set<int32_t> interior_segments;
    int32_t leaf_index = leaf_node->index;

    // traverse the path in the "interior" tree to find all segmentation regions that map to the current bounds.
    node_ptr<T> edge_low = interior_tree[leaf_node->parent->index];
    node_ptr<T> edge_high = edge_low->parent;
    int32_t last_edge_segment = -1;

    while( edge_low->index != leaf_index ){

        for( auto segment : edge_low->segments ){
            interior_segments.insert(segment);
        }

        edge_low = edge_high;

        if( edge_high->parent != nullptr ){
            edge_high = edge_high->parent;
        }

    }

    std::vector<int32_t>& final_segments = interior_tree[leaf_node->index]->segments;

    for( auto segment : final_segments ){
        interior_segments.insert(segment);
    }

    for( auto leaf_segment : leaf_segments ){
        for( auto interior_segment : interior_segments ){

            std::pair<int32_t, int32_t> segment_pair;

            if constexpr (join){
                segment_pair = {leaf_segment, interior_segment};
            } else {
                segment_pair = {interior_segment, leaf_segment};
            }

            // std::cout << "segment pair " << (int)(segment_pair.first) << " " << (int)(segment_pair.second);
            // std::cout << std::setprecision(10) << " " << (bounds_pair.first) << " " << bounds_pair.second << std::endl;

            // this feels like cheating.
            if( !bounds.count(segment_pair) ){
                bounds[segment_pair] = bounds_pair;
            }

        }
    }

}



// returns the index of a node that is potentially legal to prune
template<typename T>
int32_t add_ct_node(
    tree<T>& leaf_tree,
    tree<T>& interior_tree,
    tree<T>& ct,
    int32_t node_index,
    node_ptr<T>& leaf_node,
    node_ptr<T>& interior_node
){

    node_ptr<T> leaf_parent = leaf_node->parent;
    int32_t leaf_parent_index = leaf_parent->index;

    node_ptr<T> ct_parent;

    // first add parent if not already in the tree
    if( !ct.count(leaf_parent_index) ){
        ct_parent = std::make_shared<Node<T>>( leaf_parent_index, leaf_parent->value, (leaf_parent->segments)[0] );
        ct[leaf_parent_index] = ct_parent;
    } else {
        ct_parent = ct[leaf_parent_index];
    }

    node_ptr<T> ct_node;

    // Then add the new node if not already in the tree
    if( !ct.count(node_index) ){
        ct_node = std::make_shared<Node<T>>( node_index, leaf_node->value, (leaf_node->segments)[0] );
        ct[node_index] = ct_node;
    } else {
        ct_node = ct[node_index];
    }

    // now connect the nodes
    ct_node->parent = ct_parent;
    ct_parent->children.push_back(ct_node);

    // delete node from leaf tree
    leaf_node->parent = nullptr;
    std::vector<node_ptr<T>>& lpc = leaf_parent->children; // leaf parent children
    lpc.erase(std::remove(lpc.begin(), lpc.end(), leaf_node), lpc.end());

    // delete node from interior tree
    node_ptr<T> interior_parent = interior_node->parent;
    node_ptr<T> interior_child = (interior_node->children)[0]; // we assume there is only one child from the call conditions

    interior_child->parent = interior_parent;

    std::vector<int32_t>& cs = interior_child->segments; //child segments

    for( auto segment : interior_node->segments ){

        if( std::find(cs.begin(), cs.end(), segment) == cs.end() ){
            cs.push_back(segment);
        }

    }

    interior_node->children.clear();
    interior_node->parent = nullptr;

    // the root can be pruned, so we have to check if the parent is null
    if( interior_parent != nullptr ){
        std::vector<node_ptr<T>>& ipc = interior_parent->children; // interior parent children
        ipc.erase(std::remove(ipc.begin(), ipc.end(), interior_node), ipc.end());
        ipc.push_back( interior_child );
    }

    return leaf_parent_index;

}

template<typename T, bool join>
void unit_test_tree(
    tree<T>& t, bool topology = false
){
    int32_t root = -1;

    for( auto kv : t ){
        node_ptr<T> node = kv.second;

        while( node->parent != nullptr ){

            if( !compare_nodes<T, join>(node, node->parent) ){
                std::cout << "bad segment" << std::endl;
                exit(0);
            }

            bool pc_contains_self = false;
            for( int i = 0; i < node->parent->children.size(); ++i ){
                if( (node->parent->children)[i] == node ){
                    pc_contains_self = true;
                }
            }

            if( topology ){

                if( node->children.size() != 0 && node->child_towards_extremum == nullptr ){
                    std::cout << "child towards extremum not set" << std::endl;
                    std::cout << node->value << std::endl;
                    exit(0);
                }

                if( node->children.size() > 0 && node->children.size() - 1 > node->num_paired ){
                    std::cout << "bad pairing (less) " << node->index << " " << node->children.size() << " " << (int)node->num_paired << std::endl;
                    for( int i = 0; i < node->children.size(); ++i ){
                        std::cout << (int)((node->children)[i]->index) << std::endl;
                    }
                    exit(0);
                }

                if( node->children.size() > 0 && node->children.size() - 1 < node->num_paired ){
                    std::cout << "bad pairing(more) " << node->index << " " << node->children.size() << " " << (int)node->num_paired << std::endl;
                }

            }

            if( !pc_contains_self ){
                std::cout << "bad child containment" << std::endl;
                exit(0);
            }

            node = node->parent;



        }

        if( node->index != root && root != -1 ){
            std::cout << "multiple connected components" << std::endl;
            exit(0);
        }

        root = node->index;

    }
}

// returns the index of a node that is potentially legal to prune
template<typename T> 
int32_t add_ct_node_and_handle_segmentation(
    tree<T>& augmented_jt,
    tree<T>& augmented_st,
    tree<T>& ct,
    bounds_map<T>& bounds,
    int32_t node_index
){

    node_ptr<T> jt_node = augmented_jt[node_index];
    node_ptr<T> st_node = augmented_st[node_index];

    if( jt_node->children.size() == 0 ){

        add_ct_segmentation<T,true>(jt_node, augmented_st, bounds);
        return add_ct_node<T>(augmented_jt, augmented_st, ct, node_index, jt_node, st_node);

    } else {

        add_ct_segmentation<T,false>(st_node, augmented_jt, bounds);
        return add_ct_node<T>(augmented_st, augmented_jt, ct, node_index, st_node, jt_node);

    }

}



template<typename T>
CombineTreesReturn<T> combine_trees(
    tree<T>& jt, 
    tree<T>& st, 
    int32_t* jt_segmentation,
    int32_t* st_segmentation,
    UnionFind<int32_t>& jt_segmentation_merge,
    UnionFind<int32_t>& st_segmentation_merge,
    size_t num_points
){

    tree<T> augmented_jt = clone_tree(jt);
    tree<T> augmented_st = clone_tree(st);
    tree<T> ct;

    bounds_map<T> bounds;

    augment_merge_tree<T,true>(augmented_jt, st, jt_segmentation, jt_segmentation_merge);
    augment_merge_tree<T,false>(augmented_st, jt, st_segmentation, st_segmentation_merge);

    std::vector<int32_t> prunable_nodes;

    for( const auto& kv : augmented_jt ){
        int32_t index = kv.first;
        if( augmented_jt[index]->children.size() + augmented_st[index]->children.size() == 1 ){
            prunable_nodes.push_back(index);
        }
    }

    for( size_t i = 0; i < prunable_nodes.size() - 1; ++i ){

        int32_t index = prunable_nodes[i];

        int32_t possible_next_index = add_ct_node_and_handle_segmentation(augmented_jt, augmented_st, ct, bounds, index);

        if( augmented_jt[possible_next_index]->children.size() + augmented_st[possible_next_index]->children.size() == 1 ){

            prunable_nodes.push_back(possible_next_index);

        }

    }

    CombineTreesReturn<T> ct_info;
    ct_info.bounds = bounds;
    ct_info.ct = ct;

    return ct_info;

}

template<typename T>
struct ublb_vectors{
    std::vector<T> ub;
    std::vector<T> lb;
};

template<typename T>
ublb_vectors<T> get_ublb(
    tree<T>& jt,
    tree<T>& st,
    int32_t* jt_segmentation,
    int32_t* st_segmentation,
    T* sf,
    UnionFind<int32_t>& jt_segmentation_merge,
    UnionFind<int32_t>& st_segmentation_merge,
    bounds_map<T>& bounds,
    size_t num_points
){

    ublb_vectors<T> v;
    v.ub.reserve(num_points);
    v.lb.reserve(num_points);

    for( size_t i = 0; i < num_points; ++i ){

        // store critical points losslessly.
        if( jt.count(i) ){
            T value = jt[i]->value;
            v.ub.push_back(value);
            v.lb.push_back(value);
            continue;
        } else if( st.count(i) ){
            T value = st[i]->value;
            v.ub.push_back(value);
            v.lb.push_back(value);
            continue;
        }

        T value = sf[i];

        int32_t jt_segmentation_raw = jt_segmentation[i];
        int32_t st_segmentation_raw = st_segmentation[i];

        int32_t jt_segmentation_simplified = jt_segmentation_merge.find(jt_segmentation_raw);
        int32_t st_segmentation_simplified = st_segmentation_merge.find(st_segmentation_raw);

        node_ptr<T> jt_node = jt[jt_segmentation_simplified];
        node_ptr<T> st_node = st[st_segmentation_simplified];

        // while( jt_node->value > value ){
        //     jt_node = jt_node->child_towards_extremum;
        // }

        // while( st_node->value < value ){
        //     st_node = st_node->child_towards_extremum;
        // }

        std::pair<int32_t, int32_t> combined_segmentation = {jt_node->index, st_node->index};

        // std::cout << bounds.count(combined_segmentation) << std::endl;

        std::pair<T,T> ublb = bounds[combined_segmentation];

        // if( value <= ublb.first ){
        //     v.ub.push_back(ublb.first);
        // } else {
        //     node_ptr<T> extremum = st_node;
        //     while( extremum->child_towards_extremum != nullptr ){
        //         extremum = extremum->child_towards_extremum;
        //     }
        //     v.ub.push_back(extremum->value);
        // }

        // if( value >= ublb.second ){
        //     v.lb.push_back(ublb.second);
        // } else {
        //     node_ptr<T> extremum = jt_node;
        //     while( extremum->child_towards_extremum != nullptr ){
        //         extremum = extremum->child_towards_extremum;
        //     }
        //     v.lb.push_back(extremum->value);
        // }


        v.ub.push_back(std::min(jt_node->parent->value, std::max(st_node->value, value)));
        v.lb.push_back(std::max(std::min(jt_node->value, value), st_node->parent->value));

        // v.ub.push_back(jt_node->parent->value);
        // v.lb.push_back(st_node->parent->value);

        // if( value <= ublb.first ){
        //     v.ub.push_back(ublb.first);
        // } else {
        //     v.ub.push_back(value);
        // }
        //  else {
        //     std::cout << "exception" << std::endl;
        // }

        // if( value >= ublb.second ){
        //     v.lb.push_back(ublb.second);
        // } else {
        //     v.lb.push_back(value);
        // }
        // else {
        //     std::cout << "exception" << std::endl;
        // }



        // if( i == 1 ){
        //     std::cout << std::setprecision(10) << "jt node value " << jt_node->value << std::endl;
        //     std::cout << std::setprecision(10) << "current value " << value << std::endl;
        //     std::cout << std::setprecision(10) << "st node value " << st_node->value << std::endl;
        //     std::cout << std::setprecision(10) << "bounds " << ublb.first << " " << ublb.second << std::endl;
        //     std::cout << std::setprecision(10) << "interval " << (int)jt_node->index << " " << (int)st_node->index << std::endl;
        // }

        // v.ub.push_back(ublb.first);
        // v.lb.push_back(ublb.second);
    }

    return v;

}

template<typename T>
py::array tree_to_py(tree<T>& t){
    std::vector<int32_t> indices;
    std::vector<int32_t> parents;
    indices.reserve(t.size());
    parents.reserve(t.size());

    for( auto& kv : t ){
        int32_t index = kv.first;
        node_ptr<T>& node = kv.second;

        indices.push_back(index);

        if( node->parent == nullptr ){
            parents.push_back(-1);
        } else {
            parents.push_back(node->parent->index);
        }
    }

    py::array_t<int32_t> tree_numpy({
        static_cast<py::ssize_t>(indices.size()),
        static_cast<py::ssize_t>(2)
    });
    auto out = tree_numpy.template mutable_unchecked<2>();

    for( py::ssize_t row = 0; row < static_cast<py::ssize_t>(indices.size()); ++row ){
        out(row, 0) = indices[row];
        out(row, 1) = parents[row];
    }

    return tree_numpy;
}

template<typename T>
py::array array_to_py(std::vector<T>& f, size_t size_x, size_t size_y, size_t size_z){
    py::array_t<T, py::array::f_style> array_numpy({
        static_cast<py::ssize_t>(size_x),
        static_cast<py::ssize_t>(size_y),
        static_cast<py::ssize_t>(size_z)
    });

    std::copy(f.begin(), f.end(), array_numpy.mutable_data());
    return array_numpy;
}

template<typename T>
py::tuple ct_to_py(tree<T>& ct, ublb_vectors<T>& bounds_vectors, size_t size_x, size_t size_y, size_t size_z){
    py::array tree_numpy = tree_to_py(ct);
    py::array ub = array_to_py(bounds_vectors.ub, size_x, size_y, size_z);
    py::array lb = array_to_py(bounds_vectors.lb, size_x, size_y, size_z);

    return py::make_tuple(tree_numpy, ub, lb);
}

template<typename T>
py::tuple contour_tree_ublb_impl_template( 
    const py::array_t<int32_t, py::array::f_style | py::array::forcecast>& jt_points_py, 
    const py::array_t<int32_t, py::array::f_style | py::array::forcecast>& jt_segmentation_py, 
    const py::array_t<int32_t, py::array::f_style | py::array::forcecast>& st_points_py, 
    const py::array_t<int32_t, py::array::f_style | py::array::forcecast>& st_segmentation_py, 
    const py::array_t<T, py::array::f_style | py::array::forcecast>& sf_py,
    double epsilon 
){

    py::buffer_info jt_points_info = jt_points_py.request();
    py::buffer_info jt_segmentation_info = jt_segmentation_py.request();
    py::buffer_info st_points_info = st_points_py.request();
    py::buffer_info st_segmentation_info = st_segmentation_py.request();
    py::buffer_info sf_info = sf_py.request();

    int32_t* jt_points = static_cast<int32_t*>(jt_points_info.ptr);
    int32_t* jt_segmentation = static_cast<int32_t*>(jt_segmentation_info.ptr);
    int32_t* st_points = static_cast<int32_t*>(st_points_info.ptr);
    int32_t* st_segmentation = static_cast<int32_t*>(st_segmentation_info.ptr);
    T* sf = static_cast<T*>(sf_info.ptr);

    size_t jt_size = jt_points_py.shape(0);
    size_t st_size = st_points_py.shape(0);
    size_t num_points = jt_segmentation_py.size();

    size_t size_x = jt_segmentation_py.shape(0);
    size_t size_y = jt_segmentation_py.shape(1);
    size_t size_z = jt_segmentation_py.shape(2);
    tree<T> jt = extract_tree<T>(jt_points, sf, jt_size);
    tree<T> st = extract_tree<T>(st_points, sf, st_size);


    UnionFind<int32_t> jt_segmentation_merge = persistence_pairing_and_simplification<T, true>(jt, sf, jt_segmentation, num_points, epsilon);
    UnionFind<int32_t> st_segmentation_merge = persistence_pairing_and_simplification<T, false>(st, sf, st_segmentation, num_points, epsilon);

    CombineTreesReturn ct_info = combine_trees(jt, st, jt_segmentation, st_segmentation, jt_segmentation_merge, st_segmentation_merge, num_points );

    ublb_vectors<T> bounds_vectors = get_ublb(jt, st, jt_segmentation, st_segmentation, sf, 
                                              jt_segmentation_merge, st_segmentation_merge, ct_info.bounds, num_points);

    return ct_to_py(ct_info.ct, bounds_vectors, size_x, size_y, size_z);

}
