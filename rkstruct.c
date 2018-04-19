#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h>
#include <numpy/arrayobject.h>
#include <RadarKit.h>

// Wrappers
static PyObject *PyRKInit(PyObject *self, PyObject *args, PyObject *keywords) {
    RKSetWantScreenOutput(true);
    Py_INCREF(Py_None);
    import_array();
    return Py_None;
}

static PyObject *PyRKTest(PyObject *self, PyObject *args, PyObject *keywords) {
	import_array();
    PyByteArrayObject *input;
	PyArg_ParseTuple(args, "d", &input);

	//PyObject *ret = Py_BuildValue("d", 1.2);

	npy_intp dims[] = {10, 20};
	PyObject *ret = PyDict_New();
	if (ret == NULL) {
		RKLog("Error. Unable to create a new dictionary.\n");
		return NULL;
	}
	PyObject *key = Py_BuildValue("s", "Z");
	PyObject *value = PyArray_SimpleNew(2, dims, NPY_FLOAT32);
	PyDict_SetItem(ret, key, value);

	Py_DECREF(key);
	Py_DECREF(value);
    return ret;
}

static PyObject *PyRKRayParse(PyObject *self, PyObject *args, PyObject *keywords) {
    int verbose = 0;
    PyByteArrayObject *object;
    static char *keywordList[] = {"input", "verbose", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, keywords, "Y|i", keywordList, &object, &verbose)) {
        fprintf(stderr, "Nothing provided.\n");
        return NULL;
    }

    RKRay *ray = (RKRay *)object->ob_bytes;

    npy_intp dims[] = {ray->header.gateCount};

    PyObject *dataArray = PyDict_New();
    PyObject *dataObject = NULL;

    uint8_t *data = (uint8_t *)ray->data;
    if (ray->header.productList & RKProductListDisplayZ) {
        dataObject = PyArray_SimpleNewFromData(1, dims, NPY_UINT8, data);
        PyDict_SetItem(dataArray, Py_BuildValue("s", "Zi"), dataObject);
    }
    if (ray->header.productList & RKProductListDisplayV) {
        data += ray->header.gateCount;
        dataObject = PyArray_SimpleNewFromData(1, dims, NPY_UINT8, data);
        PyDict_SetItem(dataArray, Py_BuildValue("s", "Vi"), dataObject);
    }
    if (ray->header.productList & RKProductListDisplayW) {
        data += ray->header.gateCount;
        dataObject = PyArray_SimpleNewFromData(1, dims, NPY_UINT8, data);
        PyDict_SetItem(dataArray, Py_BuildValue("s", "Wi"), dataObject);
    }
    if (ray->header.productList & RKProductListDisplayD) {
        data += ray->header.gateCount;
        dataObject = PyArray_SimpleNewFromData(1, dims, NPY_UINT8, data);
        PyDict_SetItem(dataArray, Py_BuildValue("s", "Di"), dataObject);
    }
    if (ray->header.productList & RKProductListDisplayP) {
        data += ray->header.gateCount;
        dataObject = PyArray_SimpleNewFromData(1, dims, NPY_UINT8, data);
        PyDict_SetItem(dataArray, Py_BuildValue("s", "Pi"), dataObject);
    }
    if (ray->header.productList & RKProductListDisplayR) {
        data += ray->header.gateCount;
        dataObject = PyArray_SimpleNewFromData(1, dims, NPY_UINT8, data);
        PyDict_SetItem(dataArray, Py_BuildValue("s", "Ri"), dataObject);
    }
    float *fdata = (float *)data;
    if (ray->header.productList & RKProductListProductZ) {
        dataObject = PyArray_SimpleNewFromData(1, dims, NPY_FLOAT32, fdata);
        PyDict_SetItem(dataArray, Py_BuildValue("s", "Z"), dataObject);
    }
    if (ray->header.productList & RKProductListProductV) {
        fdata += ray->header.gateCount;
        dataObject = PyArray_SimpleNewFromData(1, dims, NPY_FLOAT32, fdata);
        PyDict_SetItem(dataArray, Py_BuildValue("s", "V"), dataObject);
    }
    if (ray->header.productList & RKProductListProductW) {
        fdata += ray->header.gateCount;
        dataObject = PyArray_SimpleNewFromData(1, dims, NPY_FLOAT32, fdata);
        PyDict_SetItem(dataArray, Py_BuildValue("s", "W"), dataObject);
    }
    if (ray->header.productList & RKProductListProductD) {
        fdata += ray->header.gateCount;
        dataObject = PyArray_SimpleNewFromData(1, dims, NPY_FLOAT32, fdata);
        PyDict_SetItem(dataArray, Py_BuildValue("s", "D"), dataObject);
    }
    if (ray->header.productList & RKProductListProductP) {
        fdata += ray->header.gateCount;
        dataObject = PyArray_SimpleNewFromData(1, dims, NPY_FLOAT32, fdata);
        PyDict_SetItem(dataArray, Py_BuildValue("s", "P"), dataObject);
    }
    if (ray->header.productList & RKProductListProductR) {
        fdata += ray->header.gateCount;
        dataObject = PyArray_SimpleNewFromData(1, dims, NPY_FLOAT32, fdata);
        PyDict_SetItem(dataArray, Py_BuildValue("s", "R"), dataObject);
    }

    PyObject *ret = Py_BuildValue("{s:f,s:f,s:i,s:O,s:O,s:O}",
                                  "elevation", ray->header.startElevation,
                                  "azimuth", ray->header.startAzimuth,
                                  "gateCount", ray->header.gateCount,
                                  "sweepBegin", ray->header.marker & RKMarkerSweepBegin ? Py_True : Py_False,
                                  "sweepEnd", ray->header.marker & RKMarkerSweepEnd ? Py_True : Py_False,
                                  "moments", dataArray);

    if (verbose > 1) {
        fprintf(stderr, "   \033[48;5;197;38;5;15m C-Ext \033[0m      \033[38;5;15mEL %.2f deg   AZ %.2f deg\033[0m -> %d\n",
                ray->header.startElevation, ray->header.startAzimuth, (int)ray->header.startAzimuth);
        fdata = (float *)ray->data;
        if (ray->header.productList & RKProductListProductZ) {
            fprintf(stderr, "                Z = [%5.1f %5.1f %5.1f %5.1f %5.1f %5.1f %5.1f %5.1f %5.1f %5.1f ...\n",
                    fdata[0], fdata[1], fdata[2], fdata[3], fdata[4], fdata[5], fdata[6], fdata[7], fdata[8], fdata[9]);
        }
        if (ray->header.productList & RKProductListProductV) {
            fdata += ray->header.gateCount;
            fprintf(stderr, "                V = [%5.1f %5.1f %5.1f %5.1f %5.1f %5.1f %5.1f %5.1f %5.1f %5.1f ...\n",
                    fdata[0], fdata[1], fdata[2], fdata[3], fdata[4], fdata[5], fdata[6], fdata[7], fdata[8], fdata[9]);
        }
        if (ray->header.productList & RKProductListProductW) {
            fdata += ray->header.gateCount;
            fprintf(stderr, "                W = [%5.1f %5.1f %5.1f %5.1f %5.1f %5.1f %5.1f %5.1f %5.1f %5.1f ...\n",
                    fdata[0], fdata[1], fdata[2], fdata[3], fdata[4], fdata[5], fdata[6], fdata[7], fdata[8], fdata[9]);
        }
        if (ray->header.productList & RKProductListProductD) {
            fdata += ray->header.gateCount;
            fprintf(stderr, "                D = [%5.1f %5.1f %5.1f %5.1f %5.1f %5.1f %5.1f %5.1f %5.1f %5.1f ...\n",
                    fdata[0], fdata[1], fdata[2], fdata[3], fdata[4], fdata[5], fdata[6], fdata[7], fdata[8], fdata[9]);
        }
        if (ray->header.productList & RKProductListProductP) {
            fdata += ray->header.gateCount;
            fprintf(stderr, "                P = [%5.1f %5.1f %5.1f %5.1f %5.1f %5.1f %5.1f %5.1f %5.1f %5.1f ...\n",
                    fdata[0], fdata[1], fdata[2], fdata[3], fdata[4], fdata[5], fdata[6], fdata[7], fdata[8], fdata[9]);
        }
        if (ray->header.productList & RKProductListProductR) {
            fdata += ray->header.gateCount;
            fprintf(stderr, "                R = [%5.1f %5.1f %5.1f %5.1f %5.1f %5.1f %5.1f %5.1f %5.1f %5.1f ...\n",
                    fdata[0], fdata[1], fdata[2], fdata[3], fdata[4], fdata[5], fdata[6], fdata[7], fdata[8], fdata[9]);
        }
        data = (uint8_t *)fdata;
        if (ray->header.productList & RKProductListDisplayZ) {
            fprintf(stderr, "                Zi = [%d %d %d %d %d %d %d %d %d %d ...\n",
                    data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7], data[8], data[9]);
        }
        if (ray->header.productList & RKProductListDisplayV) {
            data += ray->header.gateCount;
            fprintf(stderr, "                Vi = [%d %d %d %d %d %d %d %d %d %d ...\n",
                    data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7], data[8], data[9]);
        }
        if (ray->header.productList & RKProductListDisplayW) {
            data += ray->header.gateCount;
            fprintf(stderr, "                Wi = [%d %d %d %d %d %d %d %d %d %d ...\n",
                    data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7], data[8], data[9]);
        }
        if (ray->header.productList & RKProductListDisplayD) {
            data += ray->header.gateCount;
            fprintf(stderr, "                Di = [%d %d %d %d %d %d %d %d %d %d ...\n",
                    data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7], data[8], data[9]);
        }
        if (ray->header.productList & RKProductListDisplayP) {
            data += ray->header.gateCount;
            fprintf(stderr, "                Pi = [%d %d %d %d %d %d %d %d %d %d ...\n",
                    data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7], data[8], data[9]);
        }
        if (ray->header.productList & RKProductListDisplayR) {
            data += ray->header.gateCount;
            fprintf(stderr, "                Ri = [%d %d %d %d %d %d %d %d %d %d ...\n",
                    data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7], data[8], data[9]);
        }
    }

    return ret;
}

static PyObject *PyRKSweepParse(PyObject *self, PyObject *args, PyObject *keywords) {
	int verbose = 0;
	PyByteArrayObject *object;
	static char *keywordList[] = {"input", "verbose", NULL};
	if (!PyArg_ParseTupleAndKeywords(args, keywords, "Y|i", keywordList, &object, &verbose)) {
		fprintf(stderr, "Nothing provided.\n");
		return NULL;
	}
	RKSweepHeader *sweepHeader = (RKSweepHeader *)object->ob_bytes;
	PyObject *ret = Py_BuildValue("{s:i,s:i}",
								  "gateCount", sweepHeader->gateCount,
								  "rayCount", sweepHeader->rayCount);
	fprintf(stderr, "gateCount = %d   rayCount = %d\n", sweepHeader->gateCount, sweepHeader->rayCount);
	return ret;
}

static PyObject *PyRKTestShowColors(PyObject *self, PyObject *args, PyObject *keywords) {
    RKTestShowColors();
    Py_INCREF(Py_None);
    return Py_None;
}

static PyObject *PyRKRead(PyObject *self, PyObject *args, PyObject *keywords) {
	int p, r, k;
    int verbose = 0;
    char *filename;
	float *scratch;

	static char *keywordList[] = {"filename", "verbose", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, keywords, "s|i", keywordList, &filename, &verbose)) {
        fprintf(stderr, "Nothing provided.\n");
        return Py_None;
    }

	RKSetWantScreenOutput(true);

    // Do this before we use any Python array creation
    import_array();
    
	// Read the sweep using RadarKit
	RKSweep *sweep = RKSweepRead(filename);
	if (sweep == NULL) {
		fprintf(stderr, "No sweep.\n");
		return Py_None;
	}

	// A new dictionary for output
	PyObject *momentDic = PyDict_New();

	// Some constants
	npy_intp dims[] = {sweep->header.rayCount, sweep->header.gateCount};

	// The first ray
	RKRay *ray = RKGetRay(sweep->rayBuffer, 0);

	// Range
	scratch = (float *)malloc(sweep->header.gateCount * sizeof(float));
	for (k = 0; k < (int)sweep->header.gateCount; k++) {
		scratch[k] = (float)k * ray->header.gateSizeMeters;
	}
	PyArrayObject *range = (PyArrayObject *)PyArray_SimpleNewFromData(1, &dims[1], NPY_FLOAT32, scratch);
	PyArray_ENABLEFLAGS(range, NPY_ARRAY_OWNDATA);

	// Azimuth
	scratch = (float *)malloc(sweep->header.rayCount * sizeof(float));
	for (k = 0; k < (int)sweep->header.rayCount; k++) {
		RKRay *ray = RKGetRay(sweep->rayBuffer, k);
		scratch[k] = ray->header.startAzimuth;
	}
	PyArrayObject *azimuth = (PyArrayObject *)PyArray_SimpleNewFromData(1, dims, NPY_FLOAT32, scratch);
	PyArray_ENABLEFLAGS(azimuth, NPY_ARRAY_OWNDATA);

	// Elevation
	scratch = (float *)malloc(sweep->header.rayCount * sizeof(float));
	for (k = 0; k < (int)sweep->header.rayCount; k++) {
		RKRay *ray = RKGetRay(sweep->rayBuffer, k);
		scratch[k] = ray->header.startElevation;
	}
	PyArrayObject *elevation = (PyArrayObject *)PyArray_SimpleNewFromData(1, dims, NPY_FLOAT32, scratch);
	PyArray_ENABLEFLAGS(elevation, NPY_ARRAY_OWNDATA);

	// Some product description
	uint32_t productIndex;
	RKName name;
	RKName symbol;

	// A shadow copy of productList so we can manipulate it without affecting the original ray
	uint32_t productList = sweep->header.productList;
	int productCount = __builtin_popcount(productList);

	for (p = 0; p < productCount; p++) {
		// Get the symbol, name, unit, colormap, etc. from the product list
		r = RKGetNextProductDescription(symbol, name, NULL, NULL, &productIndex, &productList);
		if (r != RKResultSuccess) {
			fprintf(stderr, "Early return.\n");
			break;
		}

		// A scratch space for data arragement
		scratch = (float *)malloc(sweep->header.rayCount * sweep->header.gateCount * sizeof(float));
		if (scratch == NULL) {
			RKLog("Error. Unable to allocate memory.\n");
			return Py_None;
		}

		// Arrange the data in an array
		for (k = 0; k < (int)sweep->header.rayCount; k++) {
			memcpy(scratch + k * sweep->header.gateCount, RKGetFloatDataFromRay(sweep->rays[k], productIndex), sweep->header.gateCount * sizeof(float));
		}
		PyObject *value = PyArray_SimpleNewFromData(2, dims, NPY_FLOAT32, scratch);
		PyArray_ENABLEFLAGS((PyArrayObject *)value, NPY_ARRAY_OWNDATA);
		PyObject *key = Py_BuildValue("s", symbol);
		PyDict_SetItem(momentDic, key, value);
		Py_DECREF(value);
		Py_DECREF(key);
	}

    // Return dictionary
	PyObject *ret = Py_BuildValue("{s:s,s:f,s:f,s:i,s:O,s:O,s:O,s:O,s:O,s:O}",
								  "name", sweep->header.desc.name,
								  "sweepElevation", ray->header.sweepElevation,
								  "sweepAzimuth", ray->header.sweepAzimuth,
								  "gateCount", sweep->header.gateCount,
								  "sweepBegin", Py_True,
								  "sweepEnd", Py_False,
								  "elevation", elevation,
								  "azimuth", azimuth,
								  "range", range,
								  "moments", momentDic);

	Py_DECREF(elevation);
	Py_DECREF(azimuth);
	Py_DECREF(range);
	Py_DECREF(momentDic);

    RKSweepFree(sweep);

    return ret;
}

// Standard boiler plates
static PyMethodDef PyRKMethods[] = {
    {"init",       (PyCFunction)PyRKInit,           METH_VARARGS | METH_KEYWORDS, "Init module"},
    {"test",       (PyCFunction)PyRKTest,           METH_VARARGS | METH_KEYWORDS, "Test module"},
    {"parseRay",   (PyCFunction)PyRKRayParse,       METH_VARARGS | METH_KEYWORDS, "Ray parse module"},
	{"parseSweep", (PyCFunction)PyRKSweepParse,     METH_VARARGS | METH_KEYWORDS, "Sweep parse module"},
    {"showColors", (PyCFunction)PyRKTestShowColors, METH_VARARGS | METH_KEYWORDS, "Color module"},
    {"read",       (PyCFunction)PyRKRead,           METH_VARARGS | METH_KEYWORDS, "Read a sweep"},
    {NULL, NULL, 0, NULL}
};

#if defined(PyModule_Create)

// Python 3 way

static struct PyModuleDef PyRKModule = {
    PyModuleDef_HEAD_INIT,
    "rk",
    NULL,
    -1,
    PyRKMethods
};

PyMODINIT_FUNC

PyInit_rkstruct(void) {
    return PyModule_Create(&PyRKModule);
}

#else

// Python 2 way

PyMODINIT_FUNC

initrkstruct(void) {
    (void) Py_InitModule("rkstruct", PyRKMethods);
}

#endif
