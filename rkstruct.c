#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h>
#include <numpy/arrayobject.h>
#include <RadarKit.h>

#define IS_PY3    defined(PyModule_Create)

// Wrappers
static PyObject *RKStructInit(PyObject *self, PyObject *args, PyObject *keywords) {
    RKSetWantScreenOutput(true);
    Py_INCREF(Py_None);
    import_array();
    return Py_None;
}

static PyObject *RKStructTest(PyObject *self, PyObject *args, PyObject *keywords) {
    //RKShowTypeSizes();
    //Py_buffer buffer;

    //PyObject_GetBuffer(args, )
    //PyBuffer_Release(buffer)

    //Py_INCREF(Py_None);
    //return Py_None;
    //PyObject *result = Py_BuildValue("d", 1.2);

    PyByteArrayObject *object;
    PyArg_ParseTuple(args, "Y", &object);
    
    RKRay *ray = (RKRay *)object->ob_bytes;
    
    npy_intp dims[] = {ray->header.gateCount};
    uint8_t *data = RKGetUInt8DataFromRay(ray, RKProductIndexZ);
    
    fprintf(stderr, "    C-Ext:      \033[33mEL %.2f deg   AZ %.2f deg\033[0m -> %d   Zi = [%d %d %d %d %d %d %d %d %d %d ...\n",
            ray->header.startElevation, ray->header.startAzimuth, (int)ray->header.startAzimuth,
            data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7], data[8], data[9]);
    
    PyObject *dataArrayObject = PyArray_SimpleNewFromData(1, dims, NPY_UINT8, data);
    PyObject *ret = Py_BuildValue("{s:f,s:f,s:i,s:O,s:O,s:O}",
                                  "elevation", ray->header.startElevation,
                                  "azimuth", ray->header.startAzimuth,
                                  "gateCount", ray->header.gateCount,
                                  "sweepBegin", ray->header.marker & RKMarkerSweepBegin ? Py_True : Py_False,
                                  "sweepEnd", ray->header.marker & RKMarkerSweepEnd ? Py_True : Py_False,
                                  "data", dataArrayObject);
    return ret;
}

static PyObject *RKStructTestShowColors(PyObject *self, PyObject *args, PyObject *keywords) {
    RKTestShowColors();
    Py_INCREF(Py_None);
    return Py_None;
}

// Standard boiler plates
static PyMethodDef RKStructMethods[] = {
    {"init",       (PyCFunction)RKStructInit,           METH_VARARGS | METH_KEYWORDS, "Init module"},
    {"test",       (PyCFunction)RKStructTest,           METH_VARARGS | METH_KEYWORDS, "Test module"},
    {"showColors", (PyCFunction)RKStructTestShowColors, METH_VARARGS | METH_KEYWORDS, "Color module"},
    {NULL, NULL, 0, NULL}
};

#if IS_PY3

static struct PyModuleDef RKStructModule = {
    PyModuleDef_HEAD_INIT,
    "rk",
    NULL,
    -1,
    RKStructMethods
};

#endif

PyMODINIT_FUNC

#if IS_PY3

PyInit_rkstruct(void) {
    return PyModule_Create(&RKStructModule);
}

#else

initrkstruct(void) {
    (void) Py_InitModule("rkstruct", RKStructMethods);
}

#endif
