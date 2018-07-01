#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h>
#include <numpy/arrayobject.h>
#include <RadarKit.h>

#pragma mark - Life Cycle

// Some global settings
static PyObject *PyRKInit(PyObject *self, PyObject *args, PyObject *keywords) {
    RKSetWantScreenOutput(true);
    Py_INCREF(Py_None);
    return Py_None;
}

#pragma mark - Variables from RadarKit

// RadarKit version
static PyObject *PyRKVersion(void) {
    return Py_BuildValue("s", RKVersionString);
}

#pragma mark - Tests

static PyObject *PyRKTestBuildingSingleValue(void) {
    SHOW_FUNCTION_NAME
    return Py_BuildValue("d", 1.234);
}

static PyObject *PyRKTestBuildingTupleOfDictionaries(void) {
    SHOW_FUNCTION_NAME
    int k;
    float *rawData;
    PyObject *dict;
    PyObject *key;
    PyObject *value;

    PyObject *obj = PyTuple_New(2);

    npy_intp dims[] = {2, 3};

    // First dictionary {'name': 'Reflectivity', 'data': [0, 1, 2, 3, 4, 5]}
    dict = PyDict_New();
    key = Py_BuildValue("s", "name");
    value = Py_BuildValue("s", "Reflectivity");
    PyDict_SetItem(dict, key, value);
    Py_DECREF(value);
    Py_DECREF(key);
    rawData = (float *)malloc(6 * sizeof(float));
    for (k = 0; k < 6; k++) {
        rawData[k] = (float)k;
    }
    key = Py_BuildValue("s", "data");
    value = PyArray_SimpleNewFromData(2, dims, NPY_FLOAT32, rawData);
    PyArray_ENABLEFLAGS((PyArrayObject *)value, NPY_ARRAY_OWNDATA);
    PyDict_SetItem(dict, key, value);
    Py_DECREF(value);
    Py_DECREF(key);
    PyTuple_SetItem(obj, 0, dict);

    // Second dictionary {'name': 'Velocity', 'data': [0, 1, 2, 3, 4, 5]}
    dict = PyDict_New();
    key = Py_BuildValue("s", "name");
    value = Py_BuildValue("s", "Velocity");
    PyDict_SetItem(dict, key, value);
    Py_DECREF(value);
    Py_DECREF(key);
    rawData = (float *)malloc(6 * sizeof(float));
    for (k = 0; k < 6; k++) {
        rawData[k] = (float)k;
    }
    key = Py_BuildValue("s", "data");
    value = PyArray_SimpleNewFromData(2, dims, NPY_FLOAT, rawData);
    PyArray_ENABLEFLAGS((PyArrayObject *)value, NPY_ARRAY_OWNDATA);
    PyDict_SetItem(dict, key, value);
    Py_DECREF(value);
    Py_DECREF(key);
    PyTuple_SetItem(obj, 1, dict);

    return obj;
}

// Wrapper to test modules in RadarKit or PyRadarKit
static PyObject *PyRKTestByNumber(PyObject *self, PyObject *args, PyObject *keywords) {

    PyObject *argsTuple, *obj;
    int number = -1;
    char *string = NULL;
    
    import_array();

    // Item 0 is repeated in tuple in arg 1. Don't know why
    argsTuple = PyTuple_GetItem(args, 0);
    printf("PyRKTestByNumber() input type = %s (tuple expected) [%d elements]\n", argsTuple->ob_type->tp_name, (int)PyTuple_GET_SIZE(argsTuple));

    // First element as integer
    obj = PyTuple_GetItem(argsTuple, 0);
    printf("PyRKTestByNumber() ---> Type = %s (int expected)\n", obj->ob_type->tp_name);
    if (obj == NULL || strcmp(obj->ob_type->tp_name, "int")) {
        return Py_None;
    }
    number = (int)PyLong_AsLong(obj);
    printf("PyRKTestByNumber() ---> number = %d\n", number);

    // Second element as string, if exists
    if (PyTuple_GET_SIZE(argsTuple) > 1) {
        obj = PyTuple_GetItem(argsTuple, 1);
        printf("PyRKTestByNumber() ---> Type = %s (str expected) [%d elements]\n", obj->ob_type->tp_name, (int)PyTuple_GET_SIZE(obj));
        if (obj) {
            PyObject *strRepr = PyUnicode_AsEncodedString(obj, "utf-8", "~E~");
            string = PyBytes_AsString(strRepr);
        }
    }
    printf("PyRKTestByNumber() ---> string = %s\n", string == NULL ? "(null)" : string);

    // Default return is a None object
    obj = Py_None;

    switch (number) {
        case 100:
            obj = PyRKVersion();
            break;
        case 101:
            obj = PyRKTestBuildingSingleValue();
            break;
        case 102:
            obj = PyRKTestBuildingTupleOfDictionaries();
            break;
        default:
            RKTestByNumber(number, string);
            break;
    }

    return obj;
}

static PyObject *PyRKTestByNumberHelp(PyObject *self) {
    return Py_BuildValue("s", RKTestByNumberDescription());
}

#pragma mark - Parsers

static PyObject *PyRKParseRay(PyObject *self, PyObject *args, PyObject *keywords) {
    int verbose = 0;
    PyByteArrayObject *input;
    static char *keywordList[] = {"input", "verbose", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, keywords, "O|i", keywordList, &input, &verbose)) {
        fprintf(stderr, "PyRKRayParse() -> Nothing provided.\n");
        return NULL;
    }

    RKRay *ray = (RKRay *)input->ob_bytes;
    npy_intp dims[] = {ray->header.gateCount};
    PyObject *dataDict = PyDict_New();
    PyObject *value = NULL;
    PyObject *key = NULL;
    uint8_t *u8data;
    float *f32data;

    // Display data
    u8data = (uint8_t *)ray->data;
    if (ray->header.baseMomentList & RKBaseMomentListDisplayZ) {
        key = Py_BuildValue("s", "Zi");
        value = PyArray_SimpleNewFromData(1, dims, NPY_UINT8, u8data);
        PyDict_SetItem(dataDict, key, value);
        Py_DECREF(value);
        Py_DECREF(key);
    }
    if (ray->header.baseMomentList & RKBaseMomentListDisplayV) {
        u8data += ray->header.gateCount;
        key = Py_BuildValue("s", "Vi");
        value = PyArray_SimpleNewFromData(1, dims, NPY_UINT8, u8data);
        PyDict_SetItem(dataDict, key, value);
        Py_DECREF(value);
        Py_DECREF(key);
    }
    if (ray->header.baseMomentList & RKBaseMomentListDisplayW) {
        u8data += ray->header.gateCount;
        key = Py_BuildValue("s", "Wi");
        value = PyArray_SimpleNewFromData(1, dims, NPY_UINT8, u8data);
        PyDict_SetItem(dataDict, key, value);
        Py_DECREF(value);
        Py_DECREF(key);
    }
    if (ray->header.baseMomentList & RKBaseMomentListDisplayD) {
        u8data += ray->header.gateCount;
        key = Py_BuildValue("s", "Di");
        value = PyArray_SimpleNewFromData(1, dims, NPY_UINT8, u8data);
        PyDict_SetItem(dataDict, key, value);
        Py_DECREF(value);
        Py_DECREF(key);
    }
    if (ray->header.baseMomentList & RKBaseMomentListDisplayP) {
        u8data += ray->header.gateCount;
        key = Py_BuildValue("s", "Pi");
        value = PyArray_SimpleNewFromData(1, dims, NPY_UINT8, u8data);
        PyDict_SetItem(dataDict, key, value);
        Py_DECREF(value);
        Py_DECREF(key);
    }
    if (ray->header.baseMomentList & RKBaseMomentListDisplayR) {
        u8data += ray->header.gateCount;
        key = Py_BuildValue("s", "Ri");
        value = PyArray_SimpleNewFromData(1, dims, NPY_UINT8, u8data);
        PyDict_SetItem(dataDict, key, value);
        Py_DECREF(value);
        Py_DECREF(key);
    }

    // Float data
    f32data = (float *)u8data;
    if (ray->header.baseMomentList & RKBaseMomentListProductZ) {
        value = PyArray_SimpleNewFromData(1, dims, NPY_FLOAT32, f32data);
        key = Py_BuildValue("s", "Z");
        PyDict_SetItem(dataDict, key, value);
        Py_DECREF(value);
        Py_DECREF(key);
    }
    if (ray->header.baseMomentList & RKBaseMomentListProductV) {
        f32data += ray->header.gateCount;
        key = Py_BuildValue("s", "V");
        value = PyArray_SimpleNewFromData(1, dims, NPY_FLOAT32, f32data);
        PyDict_SetItem(dataDict, key, value);
        Py_DECREF(value);
        Py_DECREF(key);
    }
    if (ray->header.baseMomentList & RKBaseMomentListProductW) {
        f32data += ray->header.gateCount;
        key = Py_BuildValue("s", "W");
        value = PyArray_SimpleNewFromData(1, dims, NPY_FLOAT32, f32data);
        PyDict_SetItem(dataDict, key, value);
        Py_DECREF(value);
        Py_DECREF(key);
    }
    if (ray->header.baseMomentList & RKBaseMomentListProductD) {
        f32data += ray->header.gateCount;
        key = Py_BuildValue("s", "D");
        value = PyArray_SimpleNewFromData(1, dims, NPY_FLOAT32, f32data);
        PyDict_SetItem(dataDict, key, value);
        Py_DECREF(value);
        Py_DECREF(key);
    }
    if (ray->header.baseMomentList & RKBaseMomentListProductP) {
        f32data += ray->header.gateCount;
        key = Py_BuildValue("s", "P");
        value = PyArray_SimpleNewFromData(1, dims, NPY_FLOAT32, f32data);
        PyDict_SetItem(dataDict, key, value);
        Py_DECREF(value);
        Py_DECREF(key);
    }
    if (ray->header.baseMomentList & RKBaseMomentListProductR) {
        f32data += ray->header.gateCount;
        key = Py_BuildValue("s", "R");
        value = PyArray_SimpleNewFromData(1, dims, NPY_FLOAT32, f32data);
        PyDict_SetItem(dataDict, key, value);
        Py_DECREF(value);
        Py_DECREF(key);
    }

    PyObject *ret = Py_BuildValue("{s:f,s:f,s:i,s:O,s:O,s:O}",
                                  "elevation", ray->header.startElevation,
                                  "azimuth", ray->header.startAzimuth,
                                  "gateCount", ray->header.gateCount,
                                  "sweepBegin", ray->header.marker & RKMarkerSweepBegin ? Py_True : Py_False,
                                  "sweepEnd", ray->header.marker & RKMarkerSweepEnd ? Py_True : Py_False,
                                  "moments", dataDict);
    
    Py_DECREF(dataDict);

    if (verbose > 2) {
        fprintf(stderr, "   \033[38;5;15;48;5;124m  RadarKit  \033[0m \033[38;5;15mEL %.2f deg   AZ %.2f deg\033[0m -> %d\n",
                ray->header.startElevation, ray->header.startAzimuth, (int)ray->header.startAzimuth);
        f32data = (float *)ray->data;
        if (ray->header.baseMomentList & RKBaseMomentListProductZ) {
            fprintf(stderr, "                Z = [%5.1f %5.1f %5.1f %5.1f %5.1f %5.1f %5.1f %5.1f %5.1f %5.1f ...\n",
                    f32data[0], f32data[1], f32data[2], f32data[3], f32data[4], f32data[5], f32data[6], f32data[7], f32data[8], f32data[9]);
        }
        if (ray->header.baseMomentList & RKBaseMomentListProductV) {
            f32data += ray->header.gateCount;
            fprintf(stderr, "                V = [%5.1f %5.1f %5.1f %5.1f %5.1f %5.1f %5.1f %5.1f %5.1f %5.1f ...\n",
                    f32data[0], f32data[1], f32data[2], f32data[3], f32data[4], f32data[5], f32data[6], f32data[7], f32data[8], f32data[9]);
        }
        if (ray->header.baseMomentList & RKBaseMomentListProductW) {
            f32data += ray->header.gateCount;
            fprintf(stderr, "                W = [%5.1f %5.1f %5.1f %5.1f %5.1f %5.1f %5.1f %5.1f %5.1f %5.1f ...\n",
                    f32data[0], f32data[1], f32data[2], f32data[3], f32data[4], f32data[5], f32data[6], f32data[7], f32data[8], f32data[9]);
        }
        if (ray->header.baseMomentList & RKBaseMomentListProductD) {
            f32data += ray->header.gateCount;
            fprintf(stderr, "                D = [%5.1f %5.1f %5.1f %5.1f %5.1f %5.1f %5.1f %5.1f %5.1f %5.1f ...\n",
                    f32data[0], f32data[1], f32data[2], f32data[3], f32data[4], f32data[5], f32data[6], f32data[7], f32data[8], f32data[9]);
        }
        if (ray->header.baseMomentList & RKBaseMomentListProductP) {
            f32data += ray->header.gateCount;
            fprintf(stderr, "                P = [%5.1f %5.1f %5.1f %5.1f %5.1f %5.1f %5.1f %5.1f %5.1f %5.1f ...\n",
                    f32data[0], f32data[1], f32data[2], f32data[3], f32data[4], f32data[5], f32data[6], f32data[7], f32data[8], f32data[9]);
        }
        if (ray->header.baseMomentList & RKBaseMomentListProductR) {
            f32data += ray->header.gateCount;
            fprintf(stderr, "                R = [%5.1f %5.1f %5.1f %5.1f %5.1f %5.1f %5.1f %5.1f %5.1f %5.1f ...\n",
                    f32data[0], f32data[1], f32data[2], f32data[3], f32data[4], f32data[5], f32data[6], f32data[7], f32data[8], f32data[9]);
        }
        u8data = (uint8_t *)f32data;
        if (ray->header.baseMomentList & RKBaseMomentListDisplayZ) {
            fprintf(stderr, "                Zi = [%d %d %d %d %d %d %d %d %d %d ...\n",
                    u8data[0], u8data[1], u8data[2], u8data[3], u8data[4], u8data[5], u8data[6], u8data[7], u8data[8], u8data[9]);
        }
        if (ray->header.baseMomentList & RKBaseMomentListDisplayV) {
            u8data += ray->header.gateCount;
            fprintf(stderr, "                Vi = [%d %d %d %d %d %d %d %d %d %d ...\n",
                    u8data[0], u8data[1], u8data[2], u8data[3], u8data[4], u8data[5], u8data[6], u8data[7], u8data[8], u8data[9]);
        }
        if (ray->header.baseMomentList & RKBaseMomentListDisplayW) {
            u8data += ray->header.gateCount;
            fprintf(stderr, "                Wi = [%d %d %d %d %d %d %d %d %d %d ...\n",
                    u8data[0], u8data[1], u8data[2], u8data[3], u8data[4], u8data[5], u8data[6], u8data[7], u8data[8], u8data[9]);
        }
        if (ray->header.baseMomentList & RKBaseMomentListDisplayD) {
            u8data += ray->header.gateCount;
            fprintf(stderr, "                Di = [%d %d %d %d %d %d %d %d %d %d ...\n",
                    u8data[0], u8data[1], u8data[2], u8data[3], u8data[4], u8data[5], u8data[6], u8data[7], u8data[8], u8data[9]);
        }
        if (ray->header.baseMomentList & RKBaseMomentListDisplayP) {
            u8data += ray->header.gateCount;
            fprintf(stderr, "                Pi = [%d %d %d %d %d %d %d %d %d %d ...\n",
                    u8data[0], u8data[1], u8data[2], u8data[3], u8data[4], u8data[5], u8data[6], u8data[7], u8data[8], u8data[9]);
        }
        if (ray->header.baseMomentList & RKBaseMomentListDisplayR) {
            u8data += ray->header.gateCount;
            fprintf(stderr, "                Ri = [%d %d %d %d %d %d %d %d %d %d ...\n",
                    u8data[0], u8data[1], u8data[2], u8data[3], u8data[4], u8data[5], u8data[6], u8data[7], u8data[8], u8data[9]);
        }
    }

    return ret;
}

static PyObject *PyRKParseSweepHeader(PyObject *self, PyObject *args, PyObject *keywords) {
    int k, r;
    int verbose = 0;
    PyByteArrayObject *object;
    static char *keywordList[] = {"input", "verbose", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, keywords, "O|i", keywordList, &object, &verbose)) {
        fprintf(stderr, "PyRKSweepHeaderParse() -> Nothing provided.\n");
        return NULL;
    }

    // Type cast it to RadarKit's sweep header
    RKSweepHeader *sweepHeader = (RKSweepHeader *)object->ob_bytes;

    RKName name;
    RKName symbol;
    RKBaseMomentIndex index;
    RKBaseMomentList list = sweepHeader->baseMomentList;
    int count = __builtin_popcount(list);

    // A list of base moment symbols from the sweep header
    //PyObject *moments = PyList_New(count);
    PyObject *moments = PyTuple_New(count);

    for (k = 0; k < count; k++) {
        // Get the symbol, name, unit, colormap, etc. from the product list
        r = RKGetNextProductDescription(symbol, name, NULL, NULL, &index, &list);
        if (r != RKResultSuccess) {
            fprintf(stderr, "Early return.\n");
            break;
        }
        // List does not increase the reference count
        //PyList_SetItem(symbols, k, Py_BuildValue("s", symbol));
        PyTuple_SetItem(moments, k, Py_BuildValue("s", symbol));
    }
    PyObject *ret = Py_BuildValue("{s:s,s:K,s:i,s:i,s:f,s:f,s:f,s:d,s:d,s:f,s:O}",
                                  "name", sweepHeader->desc.name,
                                  "configId", sweepHeader->config.i,
                                  "rayCount", sweepHeader->rayCount,
                                  "gateCount", sweepHeader->gateCount,
                                  "gateSizeMeters", sweepHeader->gateSizeMeters,
                                  "sweepAzimuth", sweepHeader->config.sweepAzimuth,
                                  "sweepElevation", sweepHeader->config.sweepElevation,
                                  "latitude", sweepHeader->desc.latitude,
                                  "longitude", sweepHeader->desc.longitude,
                                  "altitude", sweepHeader->desc.radarHeight,
                                  "moments", moments);
    Py_DECREF(moments);
    
    return ret;
}

#pragma mark - Product Reader

static PyObject *PyRKRead(PyObject *self, PyObject *args, PyObject *keywords) {
    int p, r, k;
    int verbose = 0;
    char *filename;
    float *scratch;

    static char *keywordList[] = {"filename", "verbose", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, keywords, "s|i", keywordList, &filename, &verbose)) {
        fprintf(stderr, "PyRKRead() -> Nothing provided.\n");
        return Py_None;
    }

    RKSetWantScreenOutput(true);

    // Some product description
    RKName name;
    RKName symbol;
    if (RKGetSymbolFromFilename(filename, symbol)) {
        printf("symbol = %s%s%s\n", RKYellowColor, symbol, RKNoColor);
    }

    // Do this before we use any numpy array creation
    import_array();

    PyObject *ret = Py_None;

    if (!strcmp(symbol, "Z") || !strcmp(symbol, "V") || !strcmp(symbol, "W") ||
        !strcmp(symbol, "D") || !strcmp(symbol, "P") || !strcmp(symbol, "R") || !strcmp(symbol, "K")) {

        // Read the sweep using the RKSweepFileRead() function in RadarKit
        RKSweep *sweep = RKSweepFileRead(filename);
        if (sweep == NULL) {
            fprintf(stderr, "No sweep.\n");
            return Py_None;
        }

        // Some constants
        npy_intp dims[] = {sweep->header.rayCount, sweep->header.gateCount};

        // The first ray
        RKRay *ray = RKGetRay(sweep->rayBuffer, 0);

        // Range
        scratch = (float *)malloc(sweep->header.gateCount * sizeof(float));
        if (scratch == NULL) {
            RKLog("Error. Unable to allocate memory.\n");
            return Py_None;
        }
        for (k = 0; k < (int)sweep->header.gateCount; k++) {
            scratch[k] = (float)k * ray->header.gateSizeMeters;
        }
        PyArrayObject *range = (PyArrayObject *)PyArray_SimpleNewFromData(1, &dims[1], NPY_FLOAT32, scratch);
        PyArray_ENABLEFLAGS(range, NPY_ARRAY_OWNDATA);

        // Azimuth
        scratch = (float *)malloc(sweep->header.rayCount * sizeof(float));
        if (scratch == NULL) {
            RKLog("Error. Unable to allocate memory.\n");
            return Py_None;
        }
        for (k = 0; k < (int)sweep->header.rayCount; k++) {
            RKRay *ray = RKGetRay(sweep->rayBuffer, k);
            scratch[k] = ray->header.startAzimuth;
        }
        PyArrayObject *azimuth = (PyArrayObject *)PyArray_SimpleNewFromData(1, dims, NPY_FLOAT32, scratch);
        PyArray_ENABLEFLAGS(azimuth, NPY_ARRAY_OWNDATA);

        // Elevation
        scratch = (float *)malloc(sweep->header.rayCount * sizeof(float));
        if (scratch == NULL) {
            RKLog("Error. Unable to allocate memory.\n");
            return Py_None;
        }
        for (k = 0; k < (int)sweep->header.rayCount; k++) {
            RKRay *ray = RKGetRay(sweep->rayBuffer, k);
            scratch[k] = ray->header.startElevation;
        }
        PyArrayObject *elevation = (PyArrayObject *)PyArray_SimpleNewFromData(1, dims, NPY_FLOAT32, scratch);
        PyArray_ENABLEFLAGS(elevation, NPY_ARRAY_OWNDATA);

        // A shadow copy of productList so we can manipulate it without affecting the original ray
        RKBaseMomentIndex index;
        RKBaseMomentList list = sweep->header.baseMomentList;
        int count = __builtin_popcount(list);

        // A new dictionary for output
        PyObject *dict;
        PyObject *moments = PyTuple_New(count);

        // Gather the base moments
        for (p = 0; p < count; p++) {
            // Get the symbol, name, unit, colormap, etc. from the product list
            r = RKGetNextProductDescription(symbol, name, NULL, NULL, &index, &list);
            if (r != RKResultSuccess) {
                fprintf(stderr, "Early return.\n");
                break;
            }

            // A scratch space for data
            scratch = (float *)malloc(sweep->header.rayCount * sweep->header.gateCount * sizeof(float));
            if (scratch == NULL) {
                RKLog("Error. Unable to allocate memory.\n");
                return Py_None;
            }

            // Arrange the data in an array
            for (k = 0; k < (int)sweep->header.rayCount; k++) {
                memcpy(scratch + k * sweep->header.gateCount, RKGetFloatDataFromRay(sweep->rays[k], index), sweep->header.gateCount * sizeof(float));
            }

            // Create a dictionary of this sweep
            PyObject *value = PyArray_SimpleNewFromData(2, dims, NPY_FLOAT32, scratch);
            PyArray_ENABLEFLAGS((PyArrayObject *)value, NPY_ARRAY_OWNDATA);
            dict = Py_BuildValue("{s:s,s:s,s:O}",
                "name", name,
                "symbol", symbol,
                "data", value);
            Py_DECREF(value);
            PyTuple_SetItem(moments, p, dict);
        }

        // Return dictionary
        ret = Py_BuildValue("{s:s,s:K,s:i,s:i,s:f,s:f,s:f,s:d,s:d,s:f,s:O,s:O,s:O,s:O,s:O,s:O}",
                            "name", sweep->header.desc.name,
                            "configId", sweep->header.config.i,
                            "rayCount", sweep->header.rayCount,
                            "gateCount", sweep->header.gateCount,
                            "sweepAzimuth", sweep->header.config.sweepAzimuth,
                            "sweepElevation", sweep->header.config.sweepElevation,
                            "gateSizeMeters", sweep->header.gateSizeMeters,
                            "latitude", sweep->header.desc.latitude,
                            "longitude", sweep->header.desc.longitude,
                            "altitude", sweep->header.desc.radarHeight,
                            "sweepBegin", Py_True,
                            "sweepEnd", Py_False,
                            "elevation", elevation,
                            "azimuth", azimuth,
                            "range", range,
                            "moments", moments);
        Py_DECREF(elevation);
        Py_DECREF(azimuth);
        Py_DECREF(range);
        Py_DECREF(moments);

        // We are done with sweep
        RKSweepFree(sweep);

    } else {

        // Read the sweep using RKProductRead() of RadarKit
        RKProduct *product = RKProductFileReaderNC(filename);
        if (product == NULL) {
            fprintf(stderr, "No product.\n");
            return Py_None;
        }

        // Some constants
        npy_intp dims[] = {product->header.rayCount, product->header.gateCount};

        // Range
        scratch = (float *)malloc(product->header.gateCount * sizeof(float));
        if (scratch == NULL) {
            RKLog("Error. Unable to allocate memory.\n");
            return Py_None;
        }
        for (k = 0; k < (int)product->header.gateCount; k++) {
            scratch[k] = (float)k * product->header.gateSizeMeters;
        }
        PyArrayObject *range = (PyArrayObject *)PyArray_SimpleNewFromData(1, &dims[1], NPY_FLOAT32, scratch);
        PyArray_ENABLEFLAGS(range, NPY_ARRAY_OWNDATA);

        // Azimuth (RKFloat --> float)
        scratch = (float *)malloc(product->header.rayCount * sizeof(float));
        if (scratch == NULL) {
            RKLog("Error. Unable to allocate memory.\n");
            return Py_None;
        }
        for (k = 0; k < (int)product->header.rayCount; k++) {
            scratch[k] = (float)product->startAzimuth[k];
        }
        PyArrayObject *azimuth = (PyArrayObject *)PyArray_SimpleNewFromData(1, dims, NPY_FLOAT32, scratch);
        PyArray_ENABLEFLAGS(azimuth, NPY_ARRAY_OWNDATA);

        // Elevation (RKFloat --> float)
        scratch = (float *)malloc(product->header.rayCount * sizeof(float));
        if (scratch == NULL) {
            RKLog("Error. Unable to allocate memory.\n");
            return Py_None;
        }
        for (k = 0; k < (int)product->header.rayCount; k++) {
            scratch[k] = (float)product->startElevation[k];
        }
        PyArrayObject *elevation = (PyArrayObject *)PyArray_SimpleNewFromData(1, dims, NPY_FLOAT32, scratch);
        PyArray_ENABLEFLAGS(elevation, NPY_ARRAY_OWNDATA);

        // A scratch space for data
        scratch = (float *)malloc(product->header.rayCount * product->header.gateCount * sizeof(float));
        if (scratch == NULL) {
            RKLog("Error. Unable to allocate memory.\n");
            return Py_None;
        }
        float *y = scratch;
        RKFloat *x = product->data;
        for (k = 0; k < (int)(product->header.rayCount * product->header.gateCount); k++) {
            *y++ = *x++;
        }
        PyObject *value = PyArray_SimpleNewFromData(2, dims, NPY_FLOAT32, scratch);
        PyArray_ENABLEFLAGS((PyArrayObject *)value, NPY_ARRAY_OWNDATA);

        // A new dictionary for output      
        PyObject *dict = Py_BuildValue("{s:s,s:s,s:s,s:O}",
            "name", product->desc.name,
            "unit", product->desc.unit,
            "symbol", product->desc.symbol,
            "data", value);
        Py_DECREF(value);

        // Tuple of dictionary
        PyObject *moments = PyTuple_New(1);
        PyTuple_SetItem(moments, 0, dict);

        // Return dictionary
        ret = Py_BuildValue("{s:s,s:K,s:i,s:i,s:f,s:f,s:f,s:d,s:d,s:f,s:O,s:O,s:O,s:O,s:O,s:O}",
                            "name", product->header.radarName,
                            "configId", product->i,
                            "rayCount", product->header.rayCount,
                            "gateCount", product->header.gateCount,
                            "sweepAzimuth", product->header.sweepAzimuth,
                            "sweepElevation", product->header.sweepElevation,
                            "gateSizeMeters", product->header.gateSizeMeters,
                            "latitude", product->header.latitude,
                            "longitude", product->header.longitude,
                            "altitude", product->header.radarHeight,
                            "sweepBegin", Py_True,
                            "sweepEnd", Py_False,
                            "elevation", elevation,
                            "azimuth", azimuth,
                            "range", range,
                            "moments", moments);
        Py_DECREF(elevation);
        Py_DECREF(azimuth);
        Py_DECREF(range);
        Py_DECREF(moments);

        // We are done with product
        RKProductFree(product);
    }
    return ret;
}

#pragma mark - C Extension Setup

// Standard boiler plates
static PyMethodDef PyRKMethods[] = {
    {"init",             (PyCFunction)PyRKInit,               METH_NOARGS                 , "Init module"},
    {"version",          (PyCFunction)PyRKVersion,            METH_NOARGS                 , "RadarKit Version"},
    {"testByNumber",     (PyCFunction)PyRKTestByNumber,       METH_VARARGS                , "Test by number"},
    {"testByNumberHelp", (PyCFunction)PyRKTestByNumberHelp,   METH_NOARGS                 , "Test by number help text"},
    {"parseRay",         (PyCFunction)PyRKParseRay,           METH_VARARGS | METH_KEYWORDS, "Ray parse module"},
    {"parseSweepHeader", (PyCFunction)PyRKParseSweepHeader,   METH_VARARGS | METH_KEYWORDS, "Sweep header parse module"},
    {"read",             (PyCFunction)PyRKRead,               METH_VARARGS | METH_KEYWORDS, "Read a sweep"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef PyRKModule = {
    PyModuleDef_HEAD_INIT,
    "rk",
    NULL,
    -1,
    PyRKMethods
};

PyMODINIT_FUNC

PyInit_rk(void) {
    import_array();
    return PyModule_Create(&PyRKModule);
}
