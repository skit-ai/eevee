#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <math.h>


static int levenshtein_edit_distance(PyObject* ref, PyObject* hyp) {
  size_t ref_size = PyList_Size(ref);
  size_t hyp_size = PyList_Size(hyp);

  if (!fmin(ref_size, hyp_size)) {
    return fmax(ref_size, hyp_size);
  }

  int* distances = malloc(ref_size * hyp_size * sizeof(int));
  int leftval, upval, diagval;

  for (int i = 0; i < ref_size; i++) {
    for (int j = 0; j < hyp_size; j++) {
      if (i + j == 0) {
        distances[0] = (PyUnicode_Compare(PyList_GetItem(ref, i), PyList_GetItem(hyp, j)) != 0);
      } else if (i == 0) {
        distances[j] = distances[j - 1] + 1;
      } else if (j == 0) {
        distances[i * hyp_size] = distances[(i - 1) * hyp_size] + 1;
      } else {
        leftval = distances[(i - 1) * hyp_size + j] + 1;
        upval = distances[i * hyp_size + j - 1] + 1;
        diagval = distances[(i - 1) * hyp_size + j - 1] + \
          (PyUnicode_Compare(PyList_GetItem(ref, i), PyList_GetItem(hyp, j)) != 0);

        distances[i * hyp_size + j] = fmin(leftval, fmin(upval, diagval));
      }
    }
  }

  int cost = distances[(hyp_size * ref_size) - 1];

  free(distances);
  return cost;
}

static PyObject* levenshtein(PyObject *self, PyObject *args) {
  PyObject* ref;
  PyObject* hyp;

  if (!PyArg_ParseTuple(args, "OO", &ref, &hyp))
    return NULL;

  if (!PyList_Check(ref) || !PyList_Check(hyp)) {
    PyErr_SetString(PyExc_TypeError, "Arguments not of type list");
    return NULL;
  }

  return PyLong_FromLong(levenshtein_edit_distance(ref, hyp));
}

static PyMethodDef LevenshteinMethods[] =
  {
   {"levenshtein", levenshtein, METH_VARARGS, "Usual levenshtein distance for two lists of unicode strings."},
   {NULL, NULL, 0, NULL}
  };

static struct PyModuleDef levenshteinmodule =
  {
   PyModuleDef_HEAD_INIT,
   "levenshtein",
   NULL,
   -1,
   LevenshteinMethods
  };

PyMODINIT_FUNC
PyInit_levenshtein(void)
{
  return PyModule_Create(&levenshteinmodule);
}
