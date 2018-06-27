#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h>
#include <numpy/arrayobject.h>
#include <RadarKit.h>

#if PY_MAJOR_VERSION >= 3
void *init_numpy() {
    import_array();
    return NULL;
}
#else
void init_numpy() {
    import_array();
}
#endif

// Wrappers
static PyObject *PyRKInit(PyObject *self, PyObject *args, PyObject *keywords) {
    RKSetWantScreenOutput(true);
    Py_INCREF(Py_None);
    return Py_None;
}

static PyObject *PyRKTest(PyObject *self, PyObject *args, PyObject *keywords) {
    double input;
    init_numpy();
    PyArg_ParseTuple(args, "d", &input);
    printf("Input as double = %f\n", input);

    PyObject *obj;

    //obj = Py_BuildValue("d", 1.234);

    // npy_intp dims[] = {2, 3};
    //  = PyDict_New();
    // if (obj == NULL) {
    //     RKLog("Error. Unable to create a new dictionary.\n");
    //     return NULL;
    // }
    // float *data = (float *)malloc(9 * sizeof(float));
    // for (int k = 0; k < 9; k++) {
    //     data[k] = (double)(k + 1);
    // }
    // PyObject *key = Py_BuildValue("s", "Key");
    // PyObject *value = PyArray_SimpleNewFromData(2, dims, NPY_FLOAT32, data);
    // // Let Python free the data created by malloc() here
    // PyArray_ENABLEFLAGS((PyArrayObject *)value, NPY_ARRAY_OWNDATA);
    // PyDict_SetItem(obj, key, value);
    // Py_DECREF(value);
    // Py_DECREF(key);

    // PyTypeObject type = PyTupleObject;
    // PyArray_Descr desc = {
    //     .typeobj = ;
    // };

    int k;
    float *rawData;
    PyObject *dict;
    PyObject *data;

    obj = PyTuple_New(2);
    
    npy_intp dims[] = {2, 3};
  
    dict = PyDict_New();
    PyDict_SetItem(dict, Py_BuildValue("s", "name"), Py_BuildValue("s", "Reflectivity"));
    rawData = (float *)malloc(6 * sizeof(float));
    for (k = 0; k < 6; k++) {
        rawData[k] = (float)k;
    }
    data = PyArray_SimpleNewFromData(2, dims, NPY_FLOAT32, rawData);
    PyArray_ENABLEFLAGS((PyArrayObject *)data, NPY_ARRAY_OWNDATA);
    PyDict_SetItem(dict, Py_BuildValue("s", "data"), data);
    Py_DECREF(data);
    PyTuple_SetItem(obj, 0, dict);

    dict = PyDict_New();
    PyDict_SetItem(dict, Py_BuildValue("s", "name"), Py_BuildValue("s", "Velocity"));
    rawData = (float *)malloc(6 * sizeof(float));
    for (k = 0; k < 6; k++) {
        rawData[k] = (float)k;
    }

    data = PyArray_SimpleNewFromData(2, dims, NPY_FLOAT, rawData);
    PyArray_ENABLEFLAGS((PyArrayObject *)data, NPY_ARRAY_OWNDATA);
    PyDict_SetItem(dict, Py_BuildValue("s", "data"), data);
    Py_DECREF(data);
    PyTuple_SetItem(obj, 1, dict);

    return obj;
}

static PyObject *PyRKRayParse(PyObject *self, PyObject *args, PyObject *keywords) {
    int verbose = 0;
    PyByteArrayObject *object;
    static char *keywordList[] = {"input", "verbose", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, keywords, "O|i", keywordList, &object, &verbose)) {
        fprintf(stderr, "PyRKRayParse() -> Nothing provided.\n");
        return NULL;
    }

    RKRay *ray = (RKRay *)object->ob_bytes;

    npy_intp dims[] = {ray->header.gateCount};

    PyObject *dataArray = PyDict_New();
    PyObject *dataObject = NULL;

    uint8_t *data = (uint8_t *)ray->data;
    if (ray->header.baseMomentList & RKBaseMomentListDisplayZ) {
        dataObject = PyArray_SimpleNewFromData(1, dims, NPY_UINT8, data);
        PyArray_ENABLEFLAGS((PyArrayObject *)dataObject, NPY_ARRAY_OWNDATA);
        PyDict_SetItem(dataArray, Py_BuildValue("s", "Zi"), dataObject);
        Py_DECREF(dataObject);
    }
    if (ray->header.baseMomentList & RKBaseMomentListDisplayV) {
        data += ray->header.gateCount;
        dataObject = PyArray_SimpleNewFromData(1, dims, NPY_UINT8, data);
        PyArray_ENABLEFLAGS((PyArrayObject *)dataObject, NPY_ARRAY_OWNDATA);
        PyDict_SetItem(dataArray, Py_BuildValue("s", "Vi"), dataObject);
        Py_DECREF(dataObject);
    }
    if (ray->header.baseMomentList & RKBaseMomentListDisplayW) {
        data += ray->header.gateCount;
        dataObject = PyArray_SimpleNewFromData(1, dims, NPY_UINT8, data);
        PyArray_ENABLEFLAGS((PyArrayObject *)dataObject, NPY_ARRAY_OWNDATA);
        PyDict_SetItem(dataArray, Py_BuildValue("s", "Wi"), dataObject);
        Py_DECREF(dataObject);
    }
    if (ray->header.baseMomentList & RKBaseMomentListDisplayD) {
        data += ray->header.gateCount;
        dataObject = PyArray_SimpleNewFromData(1, dims, NPY_UINT8, data);
        PyArray_ENABLEFLAGS((PyArrayObject *)dataObject, NPY_ARRAY_OWNDATA);
        PyDict_SetItem(dataArray, Py_BuildValue("s", "Di"), dataObject);
        Py_DECREF(dataObject);
    }
    if (ray->header.baseMomentList & RKBaseMomentListDisplayP) {
        data += ray->header.gateCount;
        dataObject = PyArray_SimpleNewFromData(1, dims, NPY_UINT8, data);
        PyArray_ENABLEFLAGS((PyArrayObject *)dataObject, NPY_ARRAY_OWNDATA);
        PyDict_SetItem(dataArray, Py_BuildValue("s", "Pi"), dataObject);
        Py_DECREF(dataObject);
    }
    if (ray->header.baseMomentList & RKBaseMomentListDisplayR) {
        data += ray->header.gateCount;
        dataObject = PyArray_SimpleNewFromData(1, dims, NPY_UINT8, data);
        PyArray_ENABLEFLAGS((PyArrayObject *)dataObject, NPY_ARRAY_OWNDATA);
        PyDict_SetItem(dataArray, Py_BuildValue("s", "Ri"), dataObject);
        Py_DECREF(dataObject);
    }
    float *fdata = (float *)data;
    if (ray->header.baseMomentList & RKBaseMomentListProductZ) {
        dataObject = PyArray_SimpleNewFromData(1, dims, NPY_FLOAT32, fdata);
        PyArray_ENABLEFLAGS((PyArrayObject *)dataObject, NPY_ARRAY_OWNDATA);
        PyDict_SetItem(dataArray, Py_BuildValue("s", "Z"), dataObject);
        Py_DECREF(dataObject);
    }
    if (ray->header.baseMomentList & RKBaseMomentListProductV) {
        fdata += ray->header.gateCount;
        dataObject = PyArray_SimpleNewFromData(1, dims, NPY_FLOAT32, fdata);
        PyArray_ENABLEFLAGS((PyArrayObject *)dataObject, NPY_ARRAY_OWNDATA);
        PyDict_SetItem(dataArray, Py_BuildValue("s", "V"), dataObject);
        Py_DECREF(dataObject);
    }
    if (ray->header.baseMomentList & RKBaseMomentListProductW) {
        fdata += ray->header.gateCount;
        dataObject = PyArray_SimpleNewFromData(1, dims, NPY_FLOAT32, fdata);
        PyArray_ENABLEFLAGS((PyArrayObject *)dataObject, NPY_ARRAY_OWNDATA);
        PyDict_SetItem(dataArray, Py_BuildValue("s", "W"), dataObject);
        Py_DECREF(dataObject);
    }
    if (ray->header.baseMomentList & RKBaseMomentListProductD) {
        fdata += ray->header.gateCount;
        dataObject = PyArray_SimpleNewFromData(1, dims, NPY_FLOAT32, fdata);
        PyArray_ENABLEFLAGS((PyArrayObject *)dataObject, NPY_ARRAY_OWNDATA);
        PyDict_SetItem(dataArray, Py_BuildValue("s", "D"), dataObject);
        Py_DECREF(dataObject);
    }
    if (ray->header.baseMomentList & RKBaseMomentListProductP) {
        fdata += ray->header.gateCount;
        dataObject = PyArray_SimpleNewFromData(1, dims, NPY_FLOAT32, fdata);
        PyArray_ENABLEFLAGS((PyArrayObject *)dataObject, NPY_ARRAY_OWNDATA);
        PyDict_SetItem(dataArray, Py_BuildValue("s", "P"), dataObject);
        Py_DECREF(dataObject);
    }
    if (ray->header.baseMomentList & RKBaseMomentListProductR) {
        fdata += ray->header.gateCount;
        dataObject = PyArray_SimpleNewFromData(1, dims, NPY_FLOAT32, fdata);
        PyArray_ENABLEFLAGS((PyArrayObject *)dataObject, NPY_ARRAY_OWNDATA);
        PyDict_SetItem(dataArray, Py_BuildValue("s", "R"), dataObject);
        Py_DECREF(dataObject);
    }

    PyObject *ret = Py_BuildValue("{s:f,s:f,s:i,s:O,s:O,s:O}",
                                  "elevation", ray->header.startElevation,
                                  "azimuth", ray->header.startAzimuth,
                                  "gateCount", ray->header.gateCount,
                                  "sweepBegin", ray->header.marker & RKMarkerSweepBegin ? Py_True : Py_False,
                                  "sweepEnd", ray->header.marker & RKMarkerSweepEnd ? Py_True : Py_False,
                                  "moments", dataArray);
    
    Py_DECREF(dataArray);

    if (verbose > 2) {
        fprintf(stderr, "   \033[38;5;15;48;5;124m  RadarKit  \033[0m \033[38;5;15mEL %.2f deg   AZ %.2f deg\033[0m -> %d\n",
                ray->header.startElevation, ray->header.startAzimuth, (int)ray->header.startAzimuth);
        fdata = (float *)ray->data;
        if (ray->header.baseMomentList & RKBaseMomentListProductZ) {
            fprintf(stderr, "                Z = [%5.1f %5.1f %5.1f %5.1f %5.1f %5.1f %5.1f %5.1f %5.1f %5.1f ...\n",
                    fdata[0], fdata[1], fdata[2], fdata[3], fdata[4], fdata[5], fdata[6], fdata[7], fdata[8], fdata[9]);
        }
        if (ray->header.baseMomentList & RKBaseMomentListProductV) {
            fdata += ray->header.gateCount;
            fprintf(stderr, "                V = [%5.1f %5.1f %5.1f %5.1f %5.1f %5.1f %5.1f %5.1f %5.1f %5.1f ...\n",
                    fdata[0], fdata[1], fdata[2], fdata[3], fdata[4], fdata[5], fdata[6], fdata[7], fdata[8], fdata[9]);
        }
        if (ray->header.baseMomentList & RKBaseMomentListProductW) {
            fdata += ray->header.gateCount;
            fprintf(stderr, "                W = [%5.1f %5.1f %5.1f %5.1f %5.1f %5.1f %5.1f %5.1f %5.1f %5.1f ...\n",
                    fdata[0], fdata[1], fdata[2], fdata[3], fdata[4], fdata[5], fdata[6], fdata[7], fdata[8], fdata[9]);
        }
        if (ray->header.baseMomentList & RKBaseMomentListProductD) {
            fdata += ray->header.gateCount;
            fprintf(stderr, "                D = [%5.1f %5.1f %5.1f %5.1f %5.1f %5.1f %5.1f %5.1f %5.1f %5.1f ...\n",
                    fdata[0], fdata[1], fdata[2], fdata[3], fdata[4], fdata[5], fdata[6], fdata[7], fdata[8], fdata[9]);
        }
        if (ray->header.baseMomentList & RKBaseMomentListProductP) {
            fdata += ray->header.gateCount;
            fprintf(stderr, "                P = [%5.1f %5.1f %5.1f %5.1f %5.1f %5.1f %5.1f %5.1f %5.1f %5.1f ...\n",
                    fdata[0], fdata[1], fdata[2], fdata[3], fdata[4], fdata[5], fdata[6], fdata[7], fdata[8], fdata[9]);
        }
        if (ray->header.baseMomentList & RKBaseMomentListProductR) {
            fdata += ray->header.gateCount;
            fprintf(stderr, "                R = [%5.1f %5.1f %5.1f %5.1f %5.1f %5.1f %5.1f %5.1f %5.1f %5.1f ...\n",
                    fdata[0], fdata[1], fdata[2], fdata[3], fdata[4], fdata[5], fdata[6], fdata[7], fdata[8], fdata[9]);
        }
        data = (uint8_t *)fdata;
        if (ray->header.baseMomentList & RKBaseMomentListDisplayZ) {
            fprintf(stderr, "                Zi = [%d %d %d %d %d %d %d %d %d %d ...\n",
                    data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7], data[8], data[9]);
        }
        if (ray->header.baseMomentList & RKBaseMomentListDisplayV) {
            data += ray->header.gateCount;
            fprintf(stderr, "                Vi = [%d %d %d %d %d %d %d %d %d %d ...\n",
                    data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7], data[8], data[9]);
        }
        if (ray->header.baseMomentList & RKBaseMomentListDisplayW) {
            data += ray->header.gateCount;
            fprintf(stderr, "                Wi = [%d %d %d %d %d %d %d %d %d %d ...\n",
                    data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7], data[8], data[9]);
        }
        if (ray->header.baseMomentList & RKBaseMomentListDisplayD) {
            data += ray->header.gateCount;
            fprintf(stderr, "                Di = [%d %d %d %d %d %d %d %d %d %d ...\n",
                    data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7], data[8], data[9]);
        }
        if (ray->header.baseMomentList & RKBaseMomentListDisplayP) {
            data += ray->header.gateCount;
            fprintf(stderr, "                Pi = [%d %d %d %d %d %d %d %d %d %d ...\n",
                    data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7], data[8], data[9]);
        }
        if (ray->header.baseMomentList & RKBaseMomentListDisplayR) {
            data += ray->header.gateCount;
            fprintf(stderr, "                Ri = [%d %d %d %d %d %d %d %d %d %d ...\n",
                    data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7], data[8], data[9]);
        }
    }

    return ret;
}

static PyObject *PyRKSweepHeaderParse(PyObject *self, PyObject *args, PyObject *keywords) {
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
    PyObject *momentList = PyList_New(count);

    for (k = 0; k < count; k++) {
        // Get the symbol, name, unit, colormap, etc. from the product list
        r = RKGetNextProductDescription(symbol, name, NULL, NULL, &index, &list);
        if (r != RKResultSuccess) {
            fprintf(stderr, "Early return.\n");
            break;
        }
        PyList_SetItem(momentList, k, Py_BuildValue("s", symbol));
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
                                  "moments", momentList);
    
    Py_DECREF(momentList);
    
    return ret;
}

static PyObject *PyRKTestTerminalColors(PyObject *self, PyObject *args, PyObject *keywords) {
    RKTestTerminalColors();
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
    init_numpy();

    PyObject *ret = Py_None;

    if (!strcmp(symbol, "Z") || !strcmp(symbol, "V") || !strcmp(symbol, "W") ||
        !strcmp(symbol, "D") || !strcmp(symbol, "P") || !strcmp(symbol, "R") || !strcmp(symbol, "K")) {

        // Read the sweep using RKSweepRead() of RadarKit
        RKSweep *sweep = RKSweepRead(filename);
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
        PyObject *tuple = PyTuple_New(count);

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
            PyTuple_SetItem(tuple, p, dict);
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
                            "moments", tuple);

        Py_DECREF(elevation);
        Py_DECREF(azimuth);
        Py_DECREF(range);
        Py_DECREF(tuple);

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
        PyObject *dict = Py_BuildValue("{s:s,s:s,s:O}",
            "name", product->desc.name,
            "unit", product->desc.unit,
            "symbol", product->desc.symbol,
            "data", value);
        Py_DECREF(value);

        // Tuple of dictionary
        PyObject *tuple = PyTuple_New(1);
        PyTuple_SetItem(tuple, 0, dict);
        //Py_DECREF(dict);

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
                            "moments", tuple);

        Py_DECREF(elevation);
        Py_DECREF(azimuth);
        Py_DECREF(range);
        Py_DECREF(tuple);

        RKProductFree(product);
    }


    return ret;
}

static PyObject *PyRKVersion(PyObject *self) {
    return Py_BuildValue("s", RKVersionString);
}

// Standard boiler plates
static PyMethodDef PyRKMethods[] = {
    {"init",             (PyCFunction)PyRKInit,               METH_VARARGS | METH_KEYWORDS, "Init module"},
    {"test",             (PyCFunction)PyRKTest,               METH_VARARGS | METH_KEYWORDS, "Test module"},
    {"parseRay",         (PyCFunction)PyRKRayParse,           METH_VARARGS | METH_KEYWORDS, "Ray parse module"},
    {"parseSweepHeader", (PyCFunction)PyRKSweepHeaderParse,   METH_VARARGS | METH_KEYWORDS, "Sweep header parse module"},
    {"showColors",       (PyCFunction)PyRKTestTerminalColors, METH_VARARGS | METH_KEYWORDS, "Color module"},
    {"read",             (PyCFunction)PyRKRead,               METH_VARARGS | METH_KEYWORDS, "Read a sweep"},
    {"version",          (PyCFunction)PyRKVersion,            METH_NOARGS                 , "RadarKit Version"},
    {NULL, NULL, 0, NULL}
};

#if PY_MAJOR_VERSION >= 3

// Python 3 way

static struct PyModuleDef PyRKModule = {
    PyModuleDef_HEAD_INIT,
    "rk",
    NULL,
    -1,
    PyRKMethods
};

PyMODINIT_FUNC

PyInit_rk(void) {
    init_numpy();
    return PyModule_Create(&PyRKModule);
}

#else

// Python 2 way

PyMODINIT_FUNC

initrk(void) {
    init_numpy();
    (void) Py_InitModule("rk", PyRKMethods);
}

#endif
