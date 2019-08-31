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
    return Py_BuildValue("s", RKVersionString());
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

    PyObject *argsList, *obj;
    int number = 0;
    int verbose = 0;
    char *string = NULL;
    
    import_array();

    static char *keywordList[] = {"number", "args", "verbose", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, keywords, "i|Oi", keywordList, &number, &argsList, &verbose)) {
        fprintf(stderr, "PyRKTestByNumber() -> Nothing provided.\n");
        return NULL;
    }
    if (verbose) {
        printf("PyRKTestByNumber() input type = %s (list expected) [%d elements]\n", argsList->ob_type->tp_name, (int)PyTuple_GET_SIZE(argsList));
    }

    // Input argument is treated as string for RadarKit
    if (PyList_GET_SIZE(argsList) > 0) {
        obj = PyList_GetItem(argsList, 0);
        if (verbose) {
            printf("PyRKTestByNumber() ---> Type = %s (str expected)\n", obj->ob_type->tp_name);
        }
        if (obj) {
            PyObject *strRepr = PyUnicode_AsEncodedString(obj, "utf-8", "~E~");
            string = PyBytes_AsString(strRepr);
        }
    }
    if (verbose) {
        printf("PyRKTestByNumber() ---> string = %s\n", string == NULL ? "(null)" : string);
    }

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
    return Py_BuildValue("s", RKTestByNumberDescription(0));
}

#pragma mark - Parsers

static PyObject *PyRKParseRay(PyObject *self, PyObject *args, PyObject *keywords) {
    int k;
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

    uint64_t momentListProductOrder[] = {
        RKBaseMomentListProductZ,
        RKBaseMomentListProductV,
        RKBaseMomentListProductW,
        RKBaseMomentListProductD,
        RKBaseMomentListProductP,
        RKBaseMomentListProductR,
        RKBaseMomentListProductK,
        RKBaseMomentListProductSh,
        RKBaseMomentListProductSv,
        RKBaseMomentListProductQ,
    };
    char productSymbols[][8] = {"Z", "V", "W", "D", "P", "R", "K", "Sh", "Sv", "Q"};
    uint64_t momentListDisplayOrder[] = {
        RKBaseMomentListDisplayZ,
        RKBaseMomentListDisplayV,
        RKBaseMomentListDisplayW,
        RKBaseMomentListDisplayD,
        RKBaseMomentListDisplayP,
        RKBaseMomentListDisplayR
    };
    char displaySymbols[][8] = {"Zi", "Vi", "Wi", "Di", "Pi", "Ri"};

    // Display data
    u8data = (uint8_t *)input->ob_bytes + sizeof(RKRayHeader);
    for (k = 0; k < (int)(sizeof(momentListDisplayOrder) / sizeof(uint64_t)); k++) {
        if (ray->header.baseMomentList & momentListDisplayOrder[k]) {
            key = Py_BuildValue("s", displaySymbols[k]);
            value = PyArray_SimpleNewFromData(1, dims, NPY_UINT8, u8data);
            PyDict_SetItem(dataDict, key, value);
            Py_DECREF(value);
            Py_DECREF(key);
            u8data += ray->header.gateCount;
        }
    }
    // Float data
    f32data = (float *)u8data;
    for (k = 0; k < (int)(sizeof(momentListProductOrder) / sizeof(uint64_t)); k++) {
        if (ray->header.baseMomentList & momentListProductOrder[k]) {
            value = PyArray_SimpleNewFromData(1, dims, NPY_FLOAT32, f32data);
            key = Py_BuildValue("s", productSymbols[k]);
            PyDict_SetItem(dataDict, key, value);
            Py_DECREF(value);
            Py_DECREF(key);
            f32data += ray->header.gateCount;
        }
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
        for (k = 0; k < (int)(sizeof(momentListProductOrder) / sizeof(uint64_t)); k++) {
            if (ray->header.baseMomentList & momentListProductOrder[k]) {
                fprintf(stderr, "               %2s = [%5.1f %5.1f %5.1f %5.1f %5.1f %5.1f %5.1f %5.1f %5.1f %5.1f ...\n",
                        productSymbols[k],
                        f32data[0], f32data[1], f32data[2], f32data[3], f32data[4], f32data[5], f32data[6], f32data[7], f32data[8], f32data[9]);
                f32data += ray->header.gateCount;
            }
        }
        u8data = (uint8_t *)f32data;
        for (k = 0; k < (int)(sizeof(momentListDisplayOrder) / sizeof(uint64_t)); k++) {
            if (ray->header.baseMomentList & RKBaseMomentListDisplayZ) {
                fprintf(stderr, "               %2s = [%d %d %d %d %d %d %d %d %d %d ...\n",
                        displaySymbols[k],
                        u8data[0], u8data[1], u8data[2], u8data[3], u8data[4], u8data[5], u8data[6], u8data[7], u8data[8], u8data[9]);
                u8data += ray->header.gateCount;
            }
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

    // A tuple of base moment symbols from the sweep header
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

static PyObject *PyRKReadProducts(PyObject *self, PyObject *args, PyObject *keywords) {
    int p, k;
    int verbose = 0;
    char *filename;
    float *scratch;

    static char *keywordList[] = {"filename", "verbose", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, keywords, "s|i", keywordList, &filename, &verbose)) {
        fprintf(stderr, "PyRKRead() -> Nothing provided.\n");
        return Py_None;
    }

    if (verbose) {
        RKSetWantScreenOutput(true);
    }

    // Some product description
    RKName symbol;
    RKGetSymbolFromFilename(filename, symbol);
    
    // Do this before we use any numpy array creation
    import_array();

    PyObject *ret = Py_None;
    
    RKLog("Using %s ... symbol = %s%s%s\n", filename, RKYellowColor, symbol, RKNoColor);
    
    RKProductCollection *collection = RKProductCollectionInitWithFilename(filename);
    if (collection == NULL) {
        RKLog("Error. RKProductCollectionInitWithFilename() returned NULL.\n");
        return Py_None;
    }
    
    // Some constants
    npy_intp dims[] = {collection->products[0].header.rayCount, collection->products[0].header.gateCount};

    // Range
    scratch = (float *)malloc(dims[1] * sizeof(float));
    if (scratch == NULL) {
        RKLog("Error. Unable to allocate memory for range.\n");
        return Py_None;
    }
    float dr = collection->products[0].header.gateSizeMeters;
    for (k = 0; k < (int)dims[1]; k++) {
        scratch[k] = (float)k * dr;
    }
    PyArrayObject *range = (PyArrayObject *)PyArray_SimpleNewFromData(1, &dims[1], NPY_FLOAT32, scratch);
    PyArray_ENABLEFLAGS(range, NPY_ARRAY_OWNDATA);

    // Azimuth
    scratch = (float *)malloc(dims[0] * sizeof(float));
    if (scratch == NULL) {
        RKLog("Error. Unable to allocate memory for azimuth.\n");
        return Py_None;
    }
    for (k = 0; k < dims[0]; k++) {
        scratch[k] = collection->products[0].startAzimuth[k];
    }
    PyArrayObject *azimuth = (PyArrayObject *)PyArray_SimpleNewFromData(1, dims, NPY_FLOAT32, scratch);
    PyArray_ENABLEFLAGS(azimuth, NPY_ARRAY_OWNDATA);
    
    // Elevation
    scratch = (float *)malloc(dims[0] * sizeof(float));
    if (scratch == NULL) {
        RKLog("Error. Unable to allocate memory for elevation.\n");
        return Py_None;
    }
    for (k = 0; k < dims[0]; k++) {
        scratch[k] = collection->products[0].startElevation[k];
    }
    PyArrayObject *elevation = (PyArrayObject *)PyArray_SimpleNewFromData(1, dims, NPY_FLOAT32, scratch);
    PyArray_ENABLEFLAGS(elevation, NPY_ARRAY_OWNDATA);

    // A new dictionary for output
    PyObject *products = PyDict_New();

    // Gather the products
    RKProduct *product = NULL;
    for (p = 0; p < (int)collection->count; p++) {
        product = &collection->products[p];
        
        // A scratch space for data
        scratch = (float *)malloc(dims[0] * dims[1] * sizeof(float));
        if (scratch == NULL) {
            RKLog("Error. Unable to allocate memory for moment data.\n");
            return Py_None;
        }
        float *y = scratch;
        RKFloat *x = product->data;
        for (k = 0; k < (int)(dims[0] * dims[1]); k++) {
            *y++ = (float)*x++;
        }
        PyObject *value = PyArray_SimpleNewFromData(2, dims, NPY_FLOAT32, scratch);
        PyArray_ENABLEFLAGS((PyArrayObject *)value, NPY_ARRAY_OWNDATA);

        // A new key for the output
        PyObject *key = Py_BuildValue("s", collection->products[p].desc.symbol);
        
        // A new dictionary for output
        PyObject *dict = Py_BuildValue("{s:s,s:s,s:s,s:O}",
                                       "name", product->desc.name,
                                       "unit", product->desc.unit,
                                       "symbol", product->desc.symbol,
                                       "data", value);
        Py_DECREF(value);

        // Add to the dictionary
        PyDict_SetItem(products, key, dict);

        Py_DECREF(key);
        Py_DECREF(dict);
    }

    // Return dictionary
    ret = Py_BuildValue("{s:s,s:K,s:i,s:i,s:f,s:f,s:f,s:d,s:d,s:f,s:K,s:K,"
                        "s:O,s:O,s:O,s:O,s:O,s:O,s:O,s:O}",
                        "name", collection->products[0].header.radarName,
                        "configId", collection->products[0].i,
                        "rayCount", dims[0],
                        "gateCount", dims[1],
                        "sweepAzimuth", product->header.sweepAzimuth,
                        "sweepElevation", product->header.sweepElevation,
                        "gateSizeMeters", product->header.gateSizeMeters,
                        "latitude", product->header.latitude,
                        "longitude", product->header.longitude,
                        "altitude", product->header.radarHeight,
                        "timeBegin", product->header.startTime,
                        "timeEnd", product->header.endTime,
                        "isPPI", product->header.isPPI ? Py_True : Py_False,
                        "isRHI", product->header.isRHI ? Py_True : Py_False,
                        "sweepBegin", Py_True,
                        "sweepEnd", Py_False,
                        "elevation", elevation,
                        "azimuth", azimuth,
                        "range", range,
                        "products", products);
    Py_DECREF(elevation);
    Py_DECREF(azimuth);
    Py_DECREF(range);
    Py_DECREF(products);

    // We are done with the product collection
    RKProductCollectionFree(collection);

    return ret;
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
        RKRay *ray = RKGetRayFromBuffer(sweep->rayBuffer, 0);

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
            RKRay *ray = RKGetRayFromBuffer(sweep->rayBuffer, k);
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
            RKRay *ray = RKGetRayFromBuffer(sweep->rayBuffer, k);
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
            Py_DECREF(dict);
        }

        // Return dictionary
        ret = Py_BuildValue("{s:s,s:K,s:i,s:i,s:f,s:f,s:f,s:d,s:d,s:f,"
                            "s:O,s:O,s:O,s:O,s:O,s:O,s:O,s:O}",
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
                            "isPPI", sweep->header.isPPI ? Py_True : Py_False,
                            "isRHI", sweep->header.isRHI ? Py_True : Py_False,
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
        RKProduct *product = RKProductFileReaderNC(filename, true);
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
        Py_DECREF(dict);

        // Return dictionary
        ret = Py_BuildValue("{s:s,s:K,s:i,s:i,s:f,s:f,s:f,s:d,s:d,s:f,s:K,s:K,"
                            "s:O,s:O,s:O,s:O,s:O,s:O,s:O,s:O}",
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
                            "timeBegin", product->header.startTime,
                            "timeEnd", product->header.endTime,
                            "isPPI", product->header.isPPI ? Py_True : Py_False,
                            "isRHI", product->header.isRHI ? Py_True : Py_False,
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

#pragma mark - Product Writer

//void populateFloat(RKProduct *products, count, PyObject *sweep, const char *label) {
//    PyObject *obj = PyDict_GetItemString(sweep, label);
//    if (obj == NULL) {
//        RKLog("Error. Expected '%s' in the supplied sweep dictionary.\n", label);
//    }
//    for (int p = 0; p < count; p++) {
//        products[p].header.sweepAzimuth = (float)PyFloat_AS_DOUBLE(obj);
//    }
//    RKLog("%s\n", RKVariableInString("sweepAzimuth", &products[0].header.sweepAzimuth, RKValueTypeFloat));
//}

static PyObject *PyRKWriteProducts(PyObject *self, PyObject *args, PyObject *keywords) {
    int i, k, p;
    int verbose = 0;
    PyObject *sweep;
    
    static char *keywordList[] = {"product", "verbose", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, keywords, "O|i", keywordList, &sweep, &verbose)) {
        fprintf(stderr, "PyRKWriteProducts() -> Nothing provided.\n");
        return Py_None;
    }

    RKSetWantScreenOutput(true);

    PyObject *obj, *obj_str;
    
    char *radarName;
    int rayCount = 0;
    int gateCount = 0;
    
    PyArrayObject *array;
    int dim;
    npy_intp *shape;
    NpyIter *iter;
    NpyIter_IterNextFunc *iternext;
    PyArray_Descr *dtype;
    float **fp;
    float f;
    double d;
    
    // Use 'products' key to get the count of products
    PyObject *py_products = PyDict_GetItemString(sweep, "products");
    if (py_products == NULL) {
        RKLog("Error. Expected 'products' in the supplied sweep dictionary.\n");
    }

    Py_ssize_t productCount = PyDict_Size(py_products);
    RKLog("Info. Found %d products\n", productCount);

    // Use the 'elevation' array to determine the number of rays
    obj = PyDict_GetItemString(sweep, "elevation");
    if (!strcmp("numpy.ndarray", Py_TYPE(obj)->tp_name)) {
        array = (PyArrayObject *)obj;
        dim = PyArray_NDIM(array);
        shape = PyArray_SHAPE(array);
        if (dim > 1) {
            RKLog("Error. Expected elevation to be a 1-d array.\n");
        }
        rayCount = shape[0];
    } else {
        RKLog("Error. Expected a numpy.ndarray from elevation.");
    }
    
    // Use the 'range' array to determine the number of gates
    obj = PyDict_GetItemString(sweep, "range");
    if (!strcmp("numpy.ndarray", Py_TYPE(obj)->tp_name)) {
        array = (PyArrayObject *)obj;
        dim = PyArray_NDIM(array);
        shape = PyArray_SHAPE(array);
        if (dim > 1) {
            RKLog("Error. Expected range to be a 1-d array.\n");
        }
        gateCount = shape[0];
    } else {
        RKLog("Error. Expected a numpy.ndarray from elevation.");
    }

    RKLog("%d products   %d x %d\n", productCount, rayCount, gateCount);
    //float *dst = (float *)malloc(rayCount * sizeof(float));

    // Allocate product array through RadarKit
    RKProduct *products;
    RKProductBufferAlloc(&products, productCount, rayCount, gateCount);

    // Top-level attributes
    obj = PyDict_GetItemString(sweep, "name");
    if (obj == NULL) {
        RKLog("Error. Expected 'name' in the supplied sweep dictionary.\n");
    }
    if (strcmp("str", Py_TYPE(obj)->tp_name)) {
        RKLog("Error. Variable 'name' should be a string.\n");
    }
    obj_str = PyUnicode_AsEncodedString(obj, "utf-8", "~E~");
    radarName = PyBytes_AS_STRING(obj_str);
    for (p = 0; p < productCount; p++) {
        strcpy(products[p].header.radarName, radarName);
    }
    RKLog("%s\n", RKVariableInString("header->radarName", products[0].header.radarName, RKValueTypeString));
    Py_XDECREF(obj_str);

    obj = PyDict_GetItemString(sweep, "configId");
    if (obj == NULL) {
        RKLog("Error. Expected 'configId' in the supplied sweep dictionary.\n");
    }
    RKIdentifier identifier = (RKIdentifier)PyLong_AsLong(obj);
    for (p = 0; p < productCount; p++) {
        products[p].i = identifier;
    }
    RKLog("%s\n", RKVariableInString("header->configId", &products[0].i, RKValueTypeIdentifier));
    
    obj = PyDict_GetItemString(sweep, "isPPI");
    if (obj == NULL) {
        RKLog("Error. Expected 'isPPI' in the supplied sweep dictionary.\n");
    }
    for (p = 0; p < productCount; p++) {
        products[p].header.isPPI = obj == Py_True ? true : false;
    }

    obj = PyDict_GetItemString(sweep, "isRHI");
    if (obj == NULL) {
        RKLog("Error. Expected 'isRHI' in the supplied sweep dictionary.\n");
    }
    for (p = 0; p < productCount; p++) {
        products[p].header.isRHI = obj == Py_True ? true : false;
    }

    obj = PyDict_GetItemString(sweep, "sweepAzimuth");
    if (obj == NULL) {
        RKLog("Error. Expected 'sweepAzimuth' in the supplied sweep dictionary.\n");
    }
    f = (float)PyFloat_AS_DOUBLE(obj);
    for (p = 0; p < productCount; p++) {
        products[p].header.sweepAzimuth = f;
    }
    RKLog("%s\n", RKVariableInString("header->sweepAzimuth", &products[0].header.sweepAzimuth, RKValueTypeFloat));

    obj = PyDict_GetItemString(sweep, "sweepElevation");
    if (obj == NULL) {
        RKLog("Error. Expected 'sweepElevation' in the supplied sweep dictionary.\n");
    }
    f = (float)PyFloat_AS_DOUBLE(obj);
    for (p = 0; p < productCount; p++) {
        products[p].header.sweepElevation = f;
    }
    RKLog("%s\n", RKVariableInString("header->sweepElevation", &products[0].header.sweepElevation, RKValueTypeFloat));

    obj = PyDict_GetItemString(sweep, "gateSizeMeters");
    if (obj == NULL) {
        RKLog("Error. Expected 'gateSizeMeters' in the supplied sweep dictionary.\n");
    }
    f = (float)PyFloat_AS_DOUBLE(obj);
    for (p = 0; p < productCount; p++) {
        products[p].header.gateSizeMeters = f;
    }
    RKLog("%s\n", RKVariableInString("header->gateSizeMeters", &products[0].header.gateSizeMeters, RKValueTypeFloat));

    obj = PyDict_GetItemString(sweep, "latitude");
    if (obj == NULL) {
        RKLog("Error. Expected 'latitude' in the supplied sweep dictionary.\n");
    }
    d = (double)PyFloat_AS_DOUBLE(obj);
    for (p = 0; p < productCount; p++) {
        products[p].header.latitude = d;
    }
    RKLog("%s\n", RKVariableInString("header->latitude", &products[0].header.latitude, RKValueTypeDouble));

    obj = PyDict_GetItemString(sweep, "longitude");
    if (obj == NULL) {
        RKLog("Error. Expected 'longitude' in the supplied sweep dictionary.\n");
    }
    d = (double)PyFloat_AS_DOUBLE(obj);
    for (p = 0; p < productCount; p++) {
        products[p].header.longitude = d;
    }
    RKLog("%s\n", RKVariableInString("header->longitude", &products[0].header.longitude, RKValueTypeDouble));

    obj = PyDict_GetItemString(sweep, "altitude");
    if (obj == NULL) {
        RKLog("Error. Expected 'altitude' in the supplied sweep dictionary.\n");
    }
    for (p = 0; p < productCount; p++) {
        products[p].header.radarHeight = (float)PyFloat_AS_DOUBLE(obj);
    }
    RKLog("%s\n", RKVariableInString("header->radarHeight", &products[0].header.radarHeight, RKValueTypeFloat));

    obj = PyDict_GetItemString(sweep, "timeBegin");
    if (obj == NULL) {
        RKLog("Error. Expected 'timeBegin' in the supplied sweep dictionary.\n");
    }
    for (p = 0; p < productCount; p++) {
        products[p].header.startTime = (time_t)PyLong_AsLong(obj);
    }
    RKLog("%s\n", RKVariableInString("header->startTime", &products[0].header.startTime, RKValueTypeLong));

    obj = PyDict_GetItemString(sweep, "timeEnd");
    if (obj == NULL) {
        RKLog("Error. Expected 'timeEnd' in the supplied sweep dictionary.\n");
    }
    for (p = 0; p < productCount; p++) {
        products[p].header.endTime = (time_t)PyLong_AsLong(obj);
    }
    RKLog("%s\n", RKVariableInString("header->endTime", &products[0].header.endTime, RKValueTypeLong));

    array = (PyArrayObject *)PyDict_GetItemString(sweep, "azimuth");
    dtype = PyArray_DescrFromType(NPY_FLOAT32);
    iter = NpyIter_New(array, NPY_ITER_READONLY, NPY_KEEPORDER, NPY_NO_CASTING, dtype);
    iternext = NpyIter_GetIterNext(iter, NULL);
    for (p = 0; p < productCount; p++) {
        fp = (float **)NpyIter_GetDataPtrArray(iter);
        i = 0;
        do {
            //printf(" %.4f", **fp);
            // Compute the end azimuth (will come back to this)
            f = **fp + 1.0f;
            if (f >= 360.0) {
                f -= 360.0f;
            } else if (f < -180.0) {
                f += 360.0f;
            }
            products[p].startAzimuth[i] = **fp;
            products[p].endAzimuth[i] = f;
            i++;
        } while (iternext(iter));
    }
    if (verbose > 1) {
        printf("azimuth = ");
        for (i = 0; i < rayCount; i++) {
            printf(" %.3f", products[0].startAzimuth[i]);
        }
        printf("\n");
    }
    
    array = (PyArrayObject *)PyDict_GetItemString(sweep, "elevation");
    dtype = PyArray_DescrFromType(NPY_FLOAT32);
    iter = NpyIter_New(array, NPY_ITER_READONLY, NPY_KEEPORDER, NPY_NO_CASTING, dtype);
    iternext = NpyIter_GetIterNext(iter, NULL);
    for (p = 0; p < productCount; p++) {
        fp = (float **) NpyIter_GetDataPtrArray(iter);
        i = 0;
        do {
            products[p].startElevation[i] = **fp;
            products[p].endElevation[i] = **fp;
            i++;
        } while (iternext(iter));
    }
    if (verbose > 1) {
        printf("elevation = ");
        for (i = 0; i < rayCount; i++) {
            printf(" %.3f", products[0].startElevation[i]);
        }
        printf("\n");
    }
    
    PyObject *key, *product;
    Py_ssize_t pos = 0;
    
    // Less rigorous checks from here on
    p = 0;
    float *dst;
    while (PyDict_Next(py_products, &pos, &key, &product)) {
        obj_str = PyUnicode_AsEncodedString(key, "utf-8", "~E~");
        RKLog("%s\n", RKVariableInString("key", PyBytes_AS_STRING(obj_str), RKValueTypeString));
        Py_XDECREF(obj_str);

        obj = PyDict_GetItemString(product, "name");
        if (obj == NULL) {
            RKLog("Error. Expected 'name' in the supplied product dictionary.\n");
        }
        if (!strcmp("str", Py_TYPE(obj)->tp_name)) {
            obj_str = PyUnicode_AsEncodedString(obj, "utf-8", "~E~");
            strcpy(products[p].desc.name, PyBytes_AS_STRING(obj_str));
            RKLog("    %s\n", RKVariableInString("name", products[p].desc.name, RKValueTypeString));
            Py_XDECREF(obj_str);
        }

        obj = PyDict_GetItemString(product, "unit");
        if (obj == NULL) {
            RKLog("Error. Expected 'unit' in the supplied product dictionary.\n");
        }
        if (!strcmp("str", Py_TYPE(obj)->tp_name)) {
            obj_str = PyUnicode_AsEncodedString(obj, "utf-8", "~E~");
            strcpy(products[p].desc.unit, PyBytes_AS_STRING(obj_str));
            RKLog("    %s\n", RKVariableInString("unit", products[p].desc.unit, RKValueTypeString));
            Py_XDECREF(obj_str);
        }
        
        obj = PyDict_GetItemString(product, "symbol");
        if (obj == NULL) {
            RKLog("Error. Expected 'symbol' in the supplied product dictionary.\n");
        }
        if (!strcmp("str", Py_TYPE(obj)->tp_name)) {
            obj_str = PyUnicode_AsEncodedString(obj, "utf-8", "~E~");
            strcpy(products[p].desc.symbol, PyBytes_AS_STRING(obj_str));
            RKLog("    %s\n", RKVariableInString("symbol", products[p].desc.symbol, RKValueTypeString));
            Py_XDECREF(obj_str);
        }
        
        array = (PyArrayObject *)PyDict_GetItemString(product, "data");
        if (array == NULL) {
            RKLog("Error. Expected 'data' in the supplied product dictionary.\n");
        }
        dim = PyArray_NDIM(array);
        shape = PyArray_SHAPE(array);
        RKLog("    ndim = %d   shape = %d, %d\n", dim, shape[0], shape[1]);
        if (shape[0] != rayCount || shape[1] != gateCount) {
            RKLog("Error. Inconsistent dimensions.  shape = %d x %d != %d x %d = rayCount x gateCount\n",
                  shape[0], shape[1], rayCount, gateCount);
        }
        dtype = PyArray_DescrFromType(NPY_FLOAT32);
        iter = NpyIter_New(array, NPY_ITER_READONLY, NPY_KEEPORDER, NPY_NO_CASTING, dtype);
        iternext = NpyIter_GetIterNext(iter, NULL);
        fp = (float **)NpyIter_GetDataPtrArray(iter);
        dst = products[p].data;
        do {
            *dst++ = **fp;
        } while (iternext(iter));
        NpyIter_Deallocate(iter);
        p++;
    }
    
    char filename[256];
    for (p = 0; p < productCount; p++) {
        k = sprintf(filename, "corrected/PX-");
        k += strftime(filename + k, 16, "%Y%m%d-%H%M%S", gmtime(&products[p].header.startTime));
        k += sprintf(filename + k, "-E%.1f-%s.nc", products[p].header.sweepElevation, products[p].desc.symbol);
        printf("%s\n", filename);
        
        RKProductFileWriterNC(&products[p], filename);
    }
    
    RKProductBufferFree(products, productCount);

    return Py_True;
}

#pragma mark - RadarKit Function Bridge

static PyObject *PyRKCountryFromCoordinate(PyObject *self, PyObject *args, PyObject *keywords) {
    static char *keywordList[] = {"latitude", "longitude", NULL};
    double latitude, longitude;
    if (!PyArg_ParseTupleAndKeywords(args, keywords, "dd", keywordList, &latitude, &longitude)) {
        fprintf(stderr, "PyRKGetCountryFromCoordinate() -> Imcomplete input.\n");
        return Py_None;
    }
    return Py_BuildValue("s", RKCountryFromPosition(latitude, longitude));
}

static PyObject *PyRKSetLogFolderAndPrefix(PyObject *self, PyObject *args, PyObject *keywords) {
    char *logFolder, *prefix;
    int verbose = 0;
    static char *keywordList[] = {"logFolder", "prefix", "verbose", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, keywords, "ss|i", keywordList, &logFolder, &prefix, &verbose)) {
        fprintf(stderr, "PyRKSetLogFolderAndPrefix() -> Nothing provided.\n");
        return Py_None;
    }
    
    strcpy(rkGlobalParameters.program, prefix);
    strcpy(rkGlobalParameters.logFolder, logFolder);
    
    if (verbose) {
        RKSetWantScreenOutput(true);
        printf("rkGlobalParameters.program = %s\n", rkGlobalParameters.program);
        printf("rkGlobalParameters.logFolder = %s\n", rkGlobalParameters.logFolder);
    }
    
    return Py_True;
}

static PyObject *PyRKSetLogFilename(PyObject *self, PyObject *args, PyObject *keywords) {
    char *filename;
    int verbose = 0;
    static char *keywordList[] = {"filename", "verbose", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, keywords, "s|i", keywordList, &filename, &verbose)) {
        fprintf(stderr, "PyRKSetLogFilename() -> Nothing provided.\n");
        return Py_None;
    }
    
    rkGlobalParameters.logFolder[0] = '\0';
    rkGlobalParameters.dailyLog = false;
    //rkGlobalParameters.logTimeOnly = true;
    strcpy(rkGlobalParameters.logfile, filename);

    if (verbose) {
        RKSetWantScreenOutput(true);
        printf("rkGlobalParameters.logFolder = %s\n", rkGlobalParameters.logFolder);
        printf("rkGlobalParameters.logTimeOnly = %d\n", rkGlobalParameters.logTimeOnly);
        printf("rkGlobalParameters.logfile = %s\n", rkGlobalParameters.logfile);
    }

    return Py_True;
}

static PyObject *PyRKSetVerbosity(PyObject *self, PyObject *args, PyObject *keywords) {
    return Py_True;
}

#pragma mark - C Extension Setup

// Standard boiler plates
static PyMethodDef PyRKMethods[] = {
    {"init"                  , (PyCFunction)PyRKInit                  , METH_NOARGS                  , "Init module"},
    {"version"               , (PyCFunction)PyRKVersion               , METH_NOARGS                  , "RadarKit Version"},
    {"testByNumber"          , (PyCFunction)PyRKTestByNumber          , METH_VARARGS | METH_KEYWORDS , "Test by number"},
    {"testByNumberHelp"      , (PyCFunction)PyRKTestByNumberHelp      , METH_NOARGS                  , "Test by number help text"},
    {"parseRay"              , (PyCFunction)PyRKParseRay              , METH_VARARGS | METH_KEYWORDS , "Ray parse module"},
    {"parseSweepHeader"      , (PyCFunction)PyRKParseSweepHeader      , METH_VARARGS | METH_KEYWORDS , "Sweep header parse module"},
    {"readOneNetCDF"         , (PyCFunction)PyRKRead                  , METH_VARARGS | METH_KEYWORDS , "Read a sweep / product"},
    {"readNetCDF"            , (PyCFunction)PyRKReadProducts          , METH_VARARGS | METH_KEYWORDS , "Read a collection products"},
    {"writeNetCDF"           , (PyCFunction)PyRKWriteProducts         , METH_VARARGS | METH_KEYWORDS , "Write a product"},
    {"countryFromCoordinate" , (PyCFunction)PyRKCountryFromCoordinate , METH_VARARGS | METH_KEYWORDS , "Country name from coordinate"},
    {"setLogFolderAndPrefix" , (PyCFunction)PyRKSetLogFolderAndPrefix , METH_VARARGS | METH_KEYWORDS , "Set log folder and prefix"},
    {"setLogFilename"        , (PyCFunction)PyRKSetLogFilename        , METH_VARARGS | METH_KEYWORDS , "Set log filename"},
    {"setVerbosity"          , (PyCFunction)PyRKSetVerbosity          , METH_VARARGS | METH_KEYWORDS , "Set verbosity level"},
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
