#pragma once

#include "sf/include.hpp"

struct Point{
    int x;
    int y;
    int z;
    
    constexpr Point operator+(const Point& other) const {
        return {x + other.x, y + other.y, z + other.z};
    }
};

constexpr Point neighbors[] = {{0, -1, -1}, {1, -1, -1}, {0, 0, -1}, {1, 0, -1}, {0, -1, 0}, {1, -1, 0}, {-1, 0, 0}, 
                               {1, 0, 0}, {-1, 1, 0}, {0, 1, 0}, {-1, 0, 1}, {0, 0, 1}, {-1, 1, 1}, {0, 1, 1}};

constexpr size_t num_neighbors = 14;

constexpr Point redo_neighbors[] = { {0, 2, 2}, {-1, 2, 2}, {-2, 2, 2}, {0, 1, 2}, {-1, 1, 2}, {-2, 1, 2}, {0, 0, 2}, 
                                     {-1, 0, 2}, {-2, 0, 2}, {0, 2, 1}, {-1, 2, 1}, {-2, 2, 1}, {1, 1, 1}, {0, 1, 1}, 
                                     {-1, 1, 1}, {-2, 1, 1}, {1, 0, 1}, {0, 0, 1}, {-1, 0, 1}, {-2, 0, 1}, {1, -1, 1}, 
                                     {0, -1, 1}, {-1, -1, 1}, {0, 2, 0}, {-1, 2, 0}, {-2, 2, 0}, {1, 1, 0}, {0, 1, 0}, 
                                     {-1, 1, 0}, {-2, 1, 0}, {2, 0, 0}, {1, 0, 0}, {-1, 0, 0}, {-2, 0, 0}, 
                                     {2, -1, 0}, {1, -1, 0}, {0, -1, 0}, {-1, -1, 0}, {2, -2, 0}, {1, -2, 0}, {0, -2, 0}, 
                                     {1, 1, -1}, {0, 1, -1}, {-1, 1, -1}, {2, 0, -1}, {1, 0, -1}, {0, 0, -1}, {-1, 0, -1}, 
                                     {2, -1, -1}, {1, -1, -1}, {0, -1, -1}, {-1, -1, -1}, {2, -2, -1}, {1, -2, -1}, {0, -2, -1}, 
                                     {2, 0, -2}, {1, 0, -2}, {0, 0, -2}, {2, -1, -2}, {1, -1, -2}, {0, -1, -2}, {2, -2, -2}, 
                                     {1, -2, -2}, {0, -2, -2}, };

constexpr size_t num_redo_neighbors = 64;



template<typename T>
class SF3D{

private:
    T* data;
    size_t size_x;
    size_t size_y;
    size_t size_z;
    size_t num_points;
    size_t x_stride;
    size_t y_stride;
    size_t z_stride;
    std::array<size_t, 14> link_strides;

public:

    size_t get_index(size_t x, size_t y, size_t z) const{
        return x * x_stride + y * y_stride + z * z_stride;
    }

    int get_index(const Point& p){
        return p.x * x_stride + p.y * y_stride + p.z * z_stride;
    }

    SF3D(T* data_, size_t size_x_, size_t size_y_, size_t size_z_){
        data = data_;
        size_x = size_x_;
        size_y = size_y_;
        size_z = size_z_;

        setup_shape();
    }

    void setup_shape(){
        num_points = size_x * size_y * size_z;
        x_stride = 1;
        y_stride = size_x;
        z_stride = size_x * size_y;
        
        for( size_t i = 0; i < num_neighbors; ++i ){
            Point neighbor = neighbors[i];
            link_strides[i] = neighbor.x * x_stride + neighbor.y * y_stride + neighbor.z * z_stride;
        }
    }

    inline T get(size_t x, size_t y, size_t z) const {
        return data[get_index(x,y,z)];
    }

    inline T get(size_t idx) const {
        return data[idx];
    }

    inline void set(size_t x, size_t y, size_t z, T value) {
        data[get_index(x,y,z)] = value;
    }

    inline void set(size_t idx, T value){
        data[idx] = value;
    }

    inline size_t get_size_x() const {
        return size_x;
    }

    inline size_t get_size_y() const {
        return size_y;
    }

    inline size_t get_size_z() const {
        return size_z;
    }

    inline size_t get_num_points() const {
        return num_points;
    }

    inline bool point_is_valid(const Point& p){
        return (p.x >= 0 && p.y >= 0 && p.z >= 0 && p.x < size_x && p.y < size_y && p.z < size_z);
    }

    inline link3d<T> get_link_3d(size_t x, size_t y, size_t z) const {
        link3d<T> link;

        size_t idx = x + y * size_x + z * size_x * size_y;

        for( size_t i = 0; i < num_neighbors; ++i ){
            Point neighbor = neighbors[i];

            if( (neighbor.x == -1 && x == 0) ||
                (neighbor.y == -1 && y == 0) ||
                (neighbor.z == -1 && z == 0) ||
                (neighbor.x == 1 && x == size_x - 1) ||
                (neighbor.y == 1 && y == size_y - 1) ||
                (neighbor.z == 1 && z == size_z - 1) ){

                link[i] = get_nan<T>();                    

            } else {
                link[i] = data[idx + neighbor.x * x_stride + neighbor.y * y_stride + neighbor.z * z_stride];
            }

        }

        return link;
    }
};
