%module node

// Add necessary symbols to generated header
%{
#define SWIG_FILE_WITH_INIT
#include "../include/node.h"
%}

// %inline %{
// typedef _NodeJIT NodeJIT;
//%}



/*  include the numpy typemaps */
%include "numpy.i"



// Process symbols in header
%include "../include/node.h"

/*  need this for correct module initialization */
%init %{
    import_array();
%}

/*  typemaps for the two arrays, the second will be modified in-place */
%apply (double* IN_ARRAY1, int DIM1) {(double * in_array, int size_in)}

/*  Wrapper for cos_doubles that massages the types */
%inline %{
    void set_data_ptr_numpy(NodeJIT* self_node, double * in_array, int size_in) {
        set_data_ptr(self_node, (void*)in_array);
    }

    void set_data_ptr_PyObject(NodeJIT* self_node, PyObject* object) {
        set_data_ptr(self_node, (void*) object);
    }
%}

