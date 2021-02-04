/**
 *
 **/
#include "node.h"



void init_node(NodeJIT* self_node) {
    (*self_node)._c_prev_p = NULL;
    (*self_node)._c_next_p = NULL;
    (*self_node)._c_data_p = NULL;
}

void set_prev_ptr(NodeJIT* self_node, NodeJIT* prev_node) {
    (*self_node)._c_prev_p = (void*)prev_node;
}

void set_next_ptr(NodeJIT* self_node, NodeJIT* next_node) {
    (*self_node)._c_next_p = (void*)next_node;
}

void set_data_ptr(NodeJIT* self_node, void* data_ptr) {
    (*self_node)._c_data_p = (void*)data_ptr;
}

void reset_prev_ptr(NodeJIT* self_node) {
    (*self_node)._c_prev_p = NULL;
}

void reset_next_ptr(NodeJIT* self_node) {
    (*self_node)._c_next_p = NULL;
}

void reset_data_ptr(NodeJIT* self_node) {
    (*self_node)._c_data_p = NULL;
}