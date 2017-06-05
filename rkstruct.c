#include <Python.h>
#include <numpy/arrayObject.h>
#include <RadarKit.h>

#define IS_PY3    defined(PyModule_Create)

// Wrappers
static PyObject *PyRKInit(PyObject *self, PyObject *args, PyObject *keywords) {
    RKSetWantScreenOutput(true);
    Py_INCREF(Py_None);
    return Py_None;
}

static PyObject *PyRKTest(PyObject *self, PyObject *args, PyObject *keywords) {
    //RKShowTypeSizes();
    Py_buffer buffer;

    //PyObject_GetBuffer(args, )
    //PyBuffer_Release(buffer)

    //Py_INCREF(Py_None);
    //return Py_None;
    PyObject *result = Py_BuildValue("d", 1.0);
    return result;
}

static PyObject *PyRKTestShowColors(PyObject *self, PyObject *args, PyObject *keywords) {
    RKTestShowColors();
    Py_INCREF(Py_None);
    return Py_None;
}

// Standard boiler plates
static PyMethodDef PyRKMethods[] = {
    {"init",       (PyCFunction)PyRKInit,           METH_VARARGS | METH_KEYWORDS, "Init module"},
    {"test",       (PyCFunction)PyRKTest,           METH_VARARGS | METH_KEYWORDS, "Test module"},
    {"showColors", (PyCFunction)PyRKTestShowColors, METH_VARARGS | METH_KEYWORDS, "Color module"},
    {NULL, NULL, 0, NULL}
};

#if IS_PY3

static struct PyModuleDef PyRKModule = {
    PyModuleDef_HEAD_INIT,
    "rk",
    NULL,
    -1,
    PyRKMethods
};

#endif

PyMODINIT_FUNC

#if IS_PY3

PyInit_rkstruct(void) {
    return PyModule_Create(&PyRKModule);
}

#else

initrkstruct(void) {
    (void) Py_InitModule("rkstruct", PyRKMethods);
}

#endif
