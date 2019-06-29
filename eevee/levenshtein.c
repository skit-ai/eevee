#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <math.h>

const int OPERATIONS = 3;

static PyObject* levenshtein_error_matrix(int cost, int *best_operation) {
  PyObject *error_matrix;
  // TODO: elegance please
  long deletion = (best_operation[1] == 0) ? best_operation[0] : 0;
  long insertion = (best_operation[1] == 1) ? best_operation[0] : 0;
  long substitution = (best_operation[1] == 2) ? best_operation[0] : 0;

  // TODO: get rid of this and get something elegant
  if (best_operation[2] == 0) {
    deletion += 1;
  } else if (best_operation[2] == 1) {
    insertion += 1;
  } else {
    substitution += 1;
  }

  error_matrix = PyDict_New();
  PyDict_SetItemString(error_matrix, "cost", PyLong_FromLong(cost));
  PyDict_SetItemString(error_matrix, "deletion", PyLong_FromLong(deletion));
  PyDict_SetItemString(error_matrix, "insertion", PyLong_FromLong(insertion));
  PyDict_SetItemString(error_matrix, "substitution", PyLong_FromLong(substitution));
  return error_matrix;
}

struct DistanceMatrix {
  int *matrix;
  int *operations;
  PyObject *ref;
  PyObject *hyp;
  size_t ref_size;
  size_t hyp_size;
  int (*cell_above)(const struct DistanceMatrix*, int row, int col);
  int (*cell_before)(const struct DistanceMatrix*, int row, int col);
  int (*cell_diag_above)(const struct DistanceMatrix*, int row, int col);
  void (*optimize)(const struct DistanceMatrix*, int row, int col);
  int (*get_cost)(const struct DistanceMatrix*);
  int *(*get_best_operation)(const struct DistanceMatrix*);
};

int cell_above(const struct DistanceMatrix *d, int row, int col) {
  return d->matrix[(row - 1) * d->hyp_size + col];
}

int cell_before(const struct DistanceMatrix *d, int row, int col) {
  return d->matrix[row * d->hyp_size + (col - 1)];
}

int cell_diag_above(const struct DistanceMatrix *d, int row, int col) {
  return d->matrix[(row - 1) * d->hyp_size + (col - 1)];
}

void optimize(const struct DistanceMatrix *d, int row, int col) {
  int left_cell = cell_before(d, row, col);
  int upper_cell = cell_above(d, row, col);
  int diag_cell = cell_diag_above(d, row, col);

  if (PyUnicode_Compare(PyList_GetItem(d->ref, row), PyList_GetItem(d->hyp, col)) != 0) {
    d->matrix[row * d->hyp_size + col] = fmin(left_cell, fmin(upper_cell, diag_cell)) + 1;
  } else {
    printf("match i, j: %d %d\n", row, col);
    d->matrix[row * d->hyp_size + col] = diag_cell;
  }
}

int get_cost(const struct DistanceMatrix *d) {
  return d->matrix[d->hyp_size * d->ref_size - 1];
}

int* get_best_operation(const struct DistanceMatrix *d) {
  // cell to left
  int op_deletion = (d->hyp_size - 1) * d->ref_size - 1;

  // cell to right
  int op_insertion = d->hyp_size * (d->ref_size - 1) - 1;

  // cell above diagonally
  int op_diagonal = (d->hyp_size - 1) * (d->ref_size - 1) - 1;

  int final_cost = d->get_cost(d);
  int pre_final_cost = final_cost > 0 ? final_cost - 1 : 0;

  int row = 0;

  for (row = 0; row < OPERATIONS; row++) {
    if (pre_final_cost == d->matrix[op_diagonal]) 
    result[row] = 
  }

  // TODO: ***ing ugly
  if (pre_final_cost == d->matrix[op_diagonal]) {
    result[0] = d->matrix[op_diagonal];
    result[1] = 2;
  } else if (pre_final_cost == d->matrix[op_insertion]) {
    result[0] = d->matrix[op_insertion];
    result[1] = 1;
  } else {
    result[0] = d->matrix[op_deletion];
    result[1] = 0;
  }

  if (d->hyp_size > d->ref_size) {
    result[2] = 1;
  } else if (d->hyp_size < d->ref_size) {
    result[2] = 0;
  } else {
    result[2] = 2;
  }
  printf("operation: %d index: %d \n", result[0], result[1], result[2]);
  return result;
}

static PyObject* levenshtein_edit_distance(PyObject* ref, PyObject* hyp) {
  size_t ref_size = PyList_Size(ref);
  size_t hyp_size = PyList_Size(hyp);
  int default_operation[2] = {0, 0};

  if (!fmin(ref_size, hyp_size)) {
    return levenshtein_error_matrix(fmax(ref_size, hyp_size), &default_operation);
  }

  int* distances = (int *)malloc(ref_size * hyp_size * sizeof(int));
  int operations[] = {0, 0, 0};
  struct DistanceMatrix d = {
                             distances,
                             operations,
                             ref,
                             hyp,
                             ref_size,
                             hyp_size,
                             cell_above,
                             cell_before,
                             cell_diag_above,
                             optimize,
                             get_cost,
                             get_best_operation
  };

  for (int i = 0; i < ref_size; i++) {
    for (int j = 0; j < hyp_size; j++) {
      // TODO: Think for something nicer looking
      if (i == 0 && j == 0) {
        distances[0] = (PyUnicode_Compare(PyList_GetItem(ref, i), PyList_GetItem(hyp, j)) != 0);
      } else if (i == 0) {
        distances[j] = distances[j - 1] + 1;
      } else if (j == 0) {
        distances[i * hyp_size] = distances[(i - 1) * hyp_size] + 1;
      } else {
        d.optimize(&d, i, j);
      }
      printf("matrix[%d][%d] :: Element after:: %d\n", i, j, d.matrix[(i * hyp_size) + j]);
    }
  }

  free(distances);
  int transform_cost = d.get_cost(&d);
  int *best_operation = d.get_best_operation(&d);
  return levenshtein_error_matrix(transform_cost, best_operation);
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

  return levenshtein_edit_distance(ref, hyp);
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
