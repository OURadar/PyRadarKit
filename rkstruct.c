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
    PyByteArrayObject *input;

    PyObject *ret = Py_BuildValue("d", 1.2);

    PyArg_ParseTuple(args, "d", &input);

//    bool debug;
//    static char *keywordList[] = {"input", "debug", NULL};
//    if (!PyArg_ParseTupleAndKeywords(args, keywords, "d|i", keywordList, &input, &debug)) {
//        fprintf(stderr, "Nothing specified");
//        debug = false;
//    }
    return ret;
}

static PyObject *RKStructRayParse(PyObject *self, PyObject *args, PyObject *keywords) {
    int verbose = 0;
    PyByteArrayObject *object;
    static char *keywordList[] = {"input", "verbose", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, keywords, "Y|i", keywordList, &object, &verbose)) {
        fprintf(stderr, "Nothing provided.");
        return NULL;
    }

    uint8_t *data;
    RKRay *ray = (RKRay *)object->ob_bytes;
    
    npy_intp dims[] = {ray->header.gateCount};

    PyObject *dataArray = PyDict_New();
    PyObject *dataObject = NULL;

    if (ray->header.productList & RKProductListDisplayZ) {
        dataObject = PyArray_SimpleNewFromData(1, dims, NPY_UINT8, RKGetUInt8DataFromRay(ray, RKProductIndexZ));
        PyDict_SetItem(dataArray, Py_BuildValue("s", "Z"), dataObject);
    }
    if (ray->header.productList & RKProductListDisplayV) {
        dataObject = PyArray_SimpleNewFromData(1, dims, NPY_UINT8, RKGetUInt8DataFromRay(ray, RKProductIndexV));
        PyDict_SetItem(dataArray, Py_BuildValue("s", "V"), dataObject);
    }

    PyObject *ret = Py_BuildValue("{s:f,s:f,s:i,s:O,s:O,s:O}",
                                  "elevation", ray->header.startElevation,
                                  "azimuth", ray->header.startAzimuth,
                                  "gateCount", ray->header.gateCount,
                                  "sweepBegin", ray->header.marker & RKMarkerSweepBegin ? Py_True : Py_False,
                                  "sweepEnd", ray->header.marker & RKMarkerSweepEnd ? Py_True : Py_False,
                                  "data", dataArray);

    if (verbose > 1) {
        fprintf(stderr, "    C-Ext:      \033[38;5;197mEL %.2f deg   AZ %.2f deg\033[0m -> %d\n",
                ray->header.startElevation, ray->header.startAzimuth, (int)ray->header.startAzimuth);
        if (ray->header.productList & RKProductListDisplayZ) {
            data = RKGetUInt8DataFromRay(ray, RKProductIndexZ);
            fprintf(stderr, "                Zi = [%d %d %d %d %d %d %d %d %d %d ...\n",
                    data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7], data[8], data[9]);
        }
        if (ray->header.productList & RKProductListDisplayV) {
            data = RKGetUInt8DataFromRay(ray, RKProductIndexV);
            fprintf(stderr, "                Vi = [%d %d %d %d %d %d %d %d %d %d ...\n",
                    data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7], data[8], data[9]);
        }
        if (ray->header.productList & RKProductListDisplayW) {
            data = RKGetUInt8DataFromRay(ray, RKProductIndexW);
            fprintf(stderr, "                Wi = [%d %d %d %d %d %d %d %d %d %d ...\n",
                    data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7], data[8], data[9]);
        }
    }

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
    {"parse",      (PyCFunction)RKStructRayParse,       METH_VARARGS | METH_KEYWORDS, "Ray parse module"},
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
