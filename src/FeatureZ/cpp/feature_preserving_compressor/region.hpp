#pragma once

#include <unordered_set>
#include <vector>
#include <array>
#include <utility>
#include <limits>

template<typename T>
class Region{

    std::vector<ssize_t> point_list;
    std::unordered_set<ssize_t> point_set;
    const T* sf;

public:

    ssize_t min_index = -1;
    ssize_t max_index = -1;
    T min_value = std::numeric_limits<T>::infinity();
    T max_value = -std::numeric_limits<T>::infinity();

    Region(const T* sf_ = NULL){
        sf = sf_;
    }

    void add_point(ssize_t point){
        auto [it, inserted] = point_set.insert(point);
        if( inserted ){
            point_list.push_back(point);

            if( sf != NULL ){
                T value = sf[point];
                if( value < min_value ){
                    min_value = value;
                    min_index = point;
                }
                if( value > max_value ){
                    max_value = value;
                    max_index = point;
                }
            }

        }
    }

    void merge(const Region& other){
        for(int i = 0; i < other.size(); ++i){
            add_point(other[i]);
        }
    }

    Region<T> grow(ssize_t amount, ssize_t domain_size_x, ssize_t domain_size_y, ssize_t domain_size_z) const{
        Region<T> grown_region(sf);

        for( int i = 0; i < size(); ++i ){

            ssize_t point = (*this)[i];

            ssize_t coords_x = point % domain_size_x;
            ssize_t coords_y = (point / domain_size_x) % domain_size_y;
            ssize_t coords_z = (point / (domain_size_x*domain_size_y));

            ssize_t off_x_low = std::max(-amount, -coords_x);
            ssize_t off_x_high = std::min(amount, domain_size_x - coords_x - 1);
            ssize_t off_y_low = std::max(-amount, -coords_y);
            ssize_t off_y_high = std::min(amount, domain_size_y - coords_y - 1);
            ssize_t off_z_low = std::max(-amount, -coords_z);
            ssize_t off_z_high = std::min(amount, domain_size_z - coords_z - 1);

            for( ssize_t off_x = off_x_low; off_x <= off_x_high; ++off_x ){
                for( ssize_t off_y = off_y_low; off_y <= off_y_high; ++off_y ){
                    for( ssize_t off_z = off_z_low; off_z <= off_z_high; ++off_z ){

                        ssize_t new_point = point + off_x + domain_size_x*off_y + domain_size_x*domain_size_y*off_z;
                        grown_region.add_point(new_point);

                    }
                }
            }

        }

        return grown_region;
    }

    ssize_t size() const{
        return point_list.size();
    }

    const ssize_t& operator[](size_t i) const {
        return point_list[i];
    }

};