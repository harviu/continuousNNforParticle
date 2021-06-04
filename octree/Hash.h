#pragma once
#include <cstdint>
template <typename K, typename V>
class HashNode {
public:
	HashNode(const K& key, const V& value) {
		this->key = key;
		this->value = value;
		this->next = NULL;
	}
	K getKey() const {
		return key;
	}
	V getValue() const {
		return value;
	}
	HashNode* getNext() const {
		return next;
	}
	void setValue(V value) {
		this->value = value;
	}
	void setNext(HashNode* next) {
		this->next = next;
	}
private:
	K key;
	V value;
	HashNode* next;
};

template <typename K,typename V>
class HashMap {
public:
	HashMap(uint32_t table_length = 6) {
		this->table_size = 1<<table_length;
        this->table_length = table_length;
		table = new HashNode<K, V>*[table_size]();
	}
	~HashMap() {
		for (int i = 0; i < table_size; i++) {
			HashNode<K, V>* cur = table[i];
			while (cur != NULL) {
				HashNode<K, V>* prev = cur;
				cur = cur->getNext();
				delete prev;
			}
		}
		delete[] table;
	}

    bool get(const K& key, V& value) {
        uint32_t hashValue = hashfn(key);
        HashNode<K, V>* entry = table[hashValue];

        while (entry != NULL) {
            if (entry->getKey() == key) {
                value = entry->getValue();
                return true;
            }
            entry = entry->getNext();
        }
        return false;
    }

    void put(const K& key, const V& value) {
        uint32_t hashValue = hashfn(key);
        HashNode<K, V>* prev = NULL;
        HashNode<K, V>* entry = table[hashValue];

        while (entry != NULL && entry->getKey() != key) {
            prev = entry;
            entry = entry->getNext();
        }

        if (entry == NULL) {
            entry = new HashNode<K, V>(key, value);
            if (prev == NULL) {
                // insert as first bucket
                table[hashValue] = entry;
            }
            else {
                prev->setNext(entry);
            }
        }
        else {
            // just update the value
            entry->setValue(value);
        }
    }

    void remove(const K& key) {
        uint32_t hashValue = hashfn(key);
        HashNode<K, V>* prev = NULL;
        HashNode<K, V>* entry = table[hashValue];

        while (entry != NULL && entry->getKey() != key) {
            prev = entry;
            entry = entry->getNext();
        }

        if (entry == NULL) {
            // key not found
            return;
        }
        else {
            if (prev == NULL) {
                // remove first bucket of the list
                table[hashValue] = entry->getNext();
            }
            else {
                prev->setNext(entry->getNext());
            }
            delete entry;
        }
    }
private:
    uint32_t hashfn(K key) {
        return key & (~(~0 << table_length));
    }
    uint32_t table_length;
	uint32_t table_size;
	HashNode<K, V>** table;
};