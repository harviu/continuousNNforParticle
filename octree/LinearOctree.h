#pragma once
#include <cstdint>
#include "Hash.h"

template <typename T>
struct Octant {
	T* p;
	uint8_t children_mask;
	uint32_t loc;
};

template <typename T>
class LinearOctree
	// Using a hashmap to store the octree.
{
public:
	LinearOctree(int max_depth, int table_length, int leaf_size, float points[][3])
		:hmap(table_length), max_depth(max_depth), table_length(table_length) 
	{
		if (max_depth > 10) {
			throw std::invalid_argument("Max tree depth exceed the location indexing ability.");
		}
		hmap.put(1, {NULL, 0, 1,});
	}
	~LinearOctree() { 
		;
	}
private:
	Octant<T>* access(uint32_t loc) {
		Octant<T> oct;
		if (hmap.get(loc, oct)) {
			return &oct;
		}
		else {
			return NULL;
		}
	}

	HashMap<uint32_t, Octant<T>> hmap;
	int max_depth;
	int table_length;
};
