#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <vector>

struct error_stats {
  int ins_num;
  int del_num;
  int sub_num;
  int total_cost;
};

static PyObject* levenshtein_edit_distance(PyObject &ref, PyObject &hyp) {
  std::vector<error_stats> e(PyList_Size(&ref) + 1);
  std::vector<error_stats> cur_e(PyList_Size(&hyp) + 1);

  for (size_t i = 0; i < e.size(); i ++) {
    e[i].ins_num = 0;
    e[i].sub_num = 0;
    e[i].del_num = i;
    e[i].total_cost = i;
  }

  for (size_t hyp_index = 1; hyp_index <= PyList_Size(&hyp); hyp_index ++) {
    cur_e[0] = e[0];
    cur_e[0].ins_num++;
    cur_e[0].total_cost++;

    for (size_t ref_index = 1; ref_index <= PyList_Size(&ref); ref_index ++) {
      int ins_err = e[ref_index].total_cost + 1;
      int del_err = cur_e[ref_index - 1].total_cost + 1;
      int sub_err = e[ref_index - 1].total_cost;

      if (PyUnicode_Compare(PyList_GetItem(&hyp, hyp_index - 1), PyList_GetItem(&ref, ref_index - 1)) != 0)
        sub_err++;

      if (sub_err < ins_err && sub_err < del_err) {
        cur_e[ref_index] = e[ref_index - 1];

        if (PyUnicode_Compare(PyList_GetItem(&hyp, hyp_index - 1), PyList_GetItem(&ref, ref_index - 1)) != 0)
          cur_e[ref_index].sub_num++;  // substitution error should be increased

        cur_e[ref_index].total_cost = sub_err;

      } else if (del_err < ins_err) {
        cur_e[ref_index] = cur_e[ref_index - 1];
        cur_e[ref_index].total_cost = del_err;
        cur_e[ref_index].del_num++;    // deletion number is increased.

      } else {
        cur_e[ref_index] = e[ref_index];
        cur_e[ref_index].total_cost = ins_err;
        cur_e[ref_index].ins_num++;    // insertion number is increased.
      }
    }
    e = cur_e;  // alternate for the next recursion.
  }
  size_t ref_index = e.size() - 1;

  return Py_BuildValue("(OOOO)",
                       PyLong_FromLong(e[ref_index].total_cost),
                       PyLong_FromLong(e[ref_index].ins_num),
                       PyLong_FromLong(e[ref_index].del_num),
                       PyLong_FromLong(e[ref_index].sub_num)
                       );
}

const char* levenshtein_docstring =
  "Wrapper around kaldi's levenshtein.\n"
  "Take two lists of strings and return a tuple representing the following:\n"
  "(total cost, insertions, deletions, substitutions)";

static PyObject* levenshtein(PyObject *self, PyObject *args) {
  PyObject* ref;
  PyObject* hyp;

  if (!PyArg_ParseTuple(args, "OO", &ref, &hyp))
    return NULL;

  if (!PyList_Check(ref) || !PyList_Check(hyp)) {
    PyErr_SetString(PyExc_TypeError, "Arguments not of type list");
    return NULL;
  }

  return levenshtein_edit_distance(*ref, *hyp);
}

static PyMethodDef LevenshteinMethods[] =
  {
   {"levenshtein", levenshtein, METH_VARARGS, levenshtein_docstring},
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
