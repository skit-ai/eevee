#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <math.h>

const int OPERATIONS = 3;

// Read as transformation cost
const int XFM_EFFORT = 1;
const int INSERTION_IDX = 0;
const int DELETION_IDX = 1;
const int SUBSTITUTION_IDX = 2;
const int STATE_FIRST_CELL = 0;
const int STATE_FIRST_ROW = 1;
const int STATE_FIRST_COL = 2;
const int STATE_BOUNEDED = 3;
int LIST_OPERATIONS[] = {0, 0, 0};
int* DEFAULT_OPERATIONS;
const char STR_COST[] = "cost";
const char STR_DELETION[] = "deletion";
const char STR_INSERTION[] = "insertion";
const char STR_SUBSTITUTION[] = "substitution";

/** ================================================================================
 * Struct:: DistanceMatrix
 * ------------------------
 * This region contains code that provides helper utilities for calculating
 * [levenshtein's distance](https://en.wikipedia.org/wiki/Levenshtein_distance).
 * - Transforms a `reference` string to a `hypothesis` string and returns the cost
 *   along with breakdown on operations needed for the transformation.
 * - Transformation functions being:
 *    1. Deletion
 *    2. Insertion
 *    3. Substitution
 * - utilities for:
 *     - access:
 *         - cell above
 *         - cell before (to the left)
 *         - cell diagonally above
 *         - final cost
 *         - operations that do the transformation while minimizing cost.
 *         - state of the modification to be made.
 *     - side-effects:
 *         - optimize: updates the matrix for each cell with least cost of
 *           transformation
 *         - update values in the first column with insertion costs.
 *         - update values in the first row with deletion costs.
 * ================================================================================ */
struct DistanceMatrix {
  int* matrix;
  int* operations;
  PyObject* ref;
  PyObject* hyp;
  size_t ref_size;
  size_t hyp_size;

  // access:
  int (*cell_above)(const struct DistanceMatrix*, int row, int col);
  int (*cell_before)(const struct DistanceMatrix*, int row, int col);
  int (*cell_diag_above)(const struct DistanceMatrix*, int row, int col);
  int (*get_final_cost)(const struct DistanceMatrix*);
  int (*get_state)(const struct DistanceMatrix*, int row, int col);
  char** (*get_operation_order)(const struct DistanceMatrix*, int cost, int* operation_stack);

  int (*str_match)(const struct DistanceMatrix*, PyObject* r, PyObject* c);
  int* (*get_operations_to_cell)(const struct DistanceMatrix*, int row, int col, int cost);

  // side-effects:
  void (*optimize)(const struct DistanceMatrix*, int row, int col);
  void (*incr_row)(const struct DistanceMatrix*, int idx, int cost);
  void (*incr_col)(const struct DistanceMatrix*, int idx, int cost);
  void (*init)(const struct DistanceMatrix*);
  void (*prn_matrix)(const struct DistanceMatrix*);
};

int cell_above(const struct DistanceMatrix *d, int row, int col) {
  /**
   * Given a cell's (in a matrix) row and col index
   * return contents of the cell positioned
   * vertically above
   * If such a cell is not found, return -1
   */
  int index =  (row - 1) * d->hyp_size + col;
  return index > -1 ? d->matrix[index] : -1;
}

int cell_before(const struct DistanceMatrix *d, int row, int col) {
  /**
   * Given a cell's (in a matrix) row and col index
   * return contents of the cell positioned
   * to the immideate left.
   * If such a cell is not found return -1
   */
  int index = row * d->hyp_size + (col - 1);
  return index > -1 ? d->matrix[index] : -1;
}

int cell_diag_above(const struct DistanceMatrix *d, int row, int col) {
  /**
   * Given a cell's (in a matrix) row and col index
   * return contents of the cell positioned
   * diagonally above
   * If such a cell is not found return -1
   */
  int index = (row - 1) * d->hyp_size + (col - 1);
  return index > -1 ? d->matrix[index] : -1;
}

void optimize(const struct DistanceMatrix *d, int row, int col) {
  /**
   * Calculates the transformation effort needed to convert
   * the reference-substring to hypothesis-substring.
   * Cost of transformation is `XFM_EFFORT` for deletion
   * and insertion errors.
   *
   * The effort for substitution seems a bit different but
   * it is just for the special case where the characters match.
   * In the match-case we don't need to transform, hence the value
   * obtained in `chars_match` 0/1 is also the effort needed.
   */
  int left_cell_cost = cell_before(d, row, col) + XFM_EFFORT;
  int upper_cell_cost = cell_above(d, row, col) + XFM_EFFORT;

  /**
   * This function is called iteratively,
   * `r` and `h` stand for each character
   * going through the loop
   */
  PyObject* r = PyList_GetItem(d->ref, row);
  PyObject* h = PyList_GetItem(d->hyp, col);

  int match_cost = d->str_match(d, r, h) == 1 ? 0 : 1;
  int diag_cell_cost = cell_diag_above(d, row, col) + match_cost;
  d->matrix[row * d->hyp_size + col] = fmin(left_cell_cost, fmin(upper_cell_cost, diag_cell_cost));
}

int str_match(const struct DistanceMatrix *d, PyObject* r, PyObject* h) {
  /**
   * Given two `PyObject*`s return 1 if the string contents match
   * else return 0
   */
  return PyUnicode_Compare(r, h) == 0 ? 1 : 0;
}

int get_final_cost(const struct DistanceMatrix *d) {
  /**
   * The last cell of the matrix contains the cost of conversion
   */
  return d->matrix[d->hyp_size * d->ref_size - 1];
}

int* get_operations_to_cell(const struct DistanceMatrix *d, int row, int col, int cost) {
  /**
   * Given a row, column to a cell in the distance matrix and the cost
   * of reaching until that cell. tell the number of operations needed
   * for the transition.
   */
  int* operations = (int*)calloc(OPERATIONS, sizeof(int));
  operations[DELETION_IDX] = d->cell_before(d, row, col);
  operations[INSERTION_IDX] = d->cell_above(d, row, col);
  operations[SUBSTITUTION_IDX] = d->cell_diag_above(d, row, col);

  // Edge-case if cost > row and row is 0,
  // the default course would have been to account it as a deletion
  // To ensure substitution in this case unless row == -1 and cost > 0
  // where we do want the transition to be counted as deletion.
  if (row == 0 && col > 0 && cost > 1) {
    operations[SUBSTITUTION_IDX] = operations[DELETION_IDX];
    operations[DELETION_IDX] = -1;
  } else if (row == -1 && cost > 0) {
    operations[DELETION_IDX] = cost;
  }

  // Edge-case if cost > col and col is 0,
  // the default course would have been to account it as a insertion
  // To ensure substitution in this case unless row == -1 and cost > 0
  // where we do want the transition to be counted as insertion.
  if (col == 0 && row > 0 && cost > 1) {
    operations[SUBSTITUTION_IDX] = operations[INSERTION_IDX];
    operations[INSERTION_IDX] = -1;
  } else if (col == -1 && cost > 0) {
    operations[INSERTION_IDX] = cost;
  }
  return operations;
}

int best_operation(int* ops) {
  /**
   * ops is an array of integers containing number of operations
   * required for a transition. Given such an array, this function
   * filters out negative values, so that fmin doesn't helplessly return -1
   */
  if (ops[SUBSTITUTION_IDX] == -1 && ops[INSERTION_IDX] == -1) {
    return ops[DELETION_IDX];
  } else if (ops[SUBSTITUTION_IDX] == -1 && ops[DELETION_IDX] == -1) {
    return ops[INSERTION_IDX];
  } else if (ops[INSERTION_IDX] == -1 && ops[DELETION_IDX] == -1) {
    return ops[SUBSTITUTION_IDX];
  } else if (ops[SUBSTITUTION_IDX] == -1) {
    return fmin(ops[INSERTION_IDX], ops[DELETION_IDX]);
  } else if (ops[INSERTION_IDX] == -1) {
    return fmin(ops[SUBSTITUTION_IDX], ops[DELETION_IDX]);
  } else if (ops[DELETION_IDX] == -1) {
    return fmin(ops[SUBSTITUTION_IDX], ops[INSERTION_IDX]);
  } else {
    return fmin(ops[SUBSTITUTION_IDX], fmin(ops[INSERTION_IDX], ops[DELETION_IDX]));
  }
}

char** get_operation_order(const struct DistanceMatrix *d, int cost, int* operation_stack) {
  /**
   * Given a DistanceMatrix, total transform cost. Find the exact number of operations
   * and the strings involved with the said operations to be able to make inferences on the
   * test results. operation_stack contains the sum of all operations. The array of strings
   * returned by this function describes one of the optimal solutions which is: for each strings
   * which operation should be performed.
   */
  int row = d->ref_size - 1;
  int col = d->hyp_size - 1;
  int* ops;
  int best_ops;
  int i, min_op_arg;
  PyObject* empty = PyUnicode_FromStringAndSize("", 1);
  PyObject* r;
  PyObject* h;
  PyObject* str_r;
  PyObject* unicode_r;
  PyObject* str_h;
  PyObject* unicode_h;
  char* bytes_r;
  char* bytes_h;
  char** operations_order = (char**) calloc(cost, sizeof(char*));

  for(i = 0; i < cost; i++) {
    operations_order[i] = (char*) calloc(500, sizeof(char));
  }

  while(cost > 0) {
    // Get value of operations required to reach the current cell
    ops = d->get_operations_to_cell(d, row, col, cost);

    // Choose the cheapest operation.
    best_ops = best_operation(ops);

    // Argmin
    for (i = SUBSTITUTION_IDX; i >= INSERTION_IDX; i--) {
      if (best_ops == ops[i] && i == SUBSTITUTION_IDX) {
        min_op_arg = i;
        break;
      } else if (best_ops == ops[i]) {
        min_op_arg = i;
      }
    }

    // -------------------------------------------------------------
    // PyObject to string manipulation
    // -------------------------------------------------------------
    r = row > -1 ? PyList_GetItem(d->ref, row) : empty;
    str_r = PyObject_Repr(r);
    unicode_r = PyUnicode_AsEncodedString(str_r, "utf-8", "~E~");
    bytes_r = PyBytes_AS_STRING(unicode_r);

    h = col > -1 ? PyList_GetItem(d->hyp, col) : empty;
    str_h = PyObject_Repr(h);
    unicode_h = PyUnicode_AsEncodedString(str_h, "utf-8", "~E~");
    bytes_h = PyBytes_AS_STRING(unicode_h);
    // -------------------------------------------------------------

    // if string `r` matches `h`
    // skip to the next cell
    if (d->str_match(d, r, h) && !d->str_match(d, r, empty)) {
      row = (row > -1) ? row - 1 : row;
      col = (col > -1) ? col - 1 : col;
      continue;
    }

    switch(min_op_arg) {
      case 0: {
        sprintf(operations_order[cost - 1], "ins %s\n", bytes_r);
        operation_stack[INSERTION_IDX] += 1;
      }; break;
      case 1: {
        sprintf(operations_order[cost - 1], "del %s\n", bytes_h);
        operation_stack[DELETION_IDX] += 1;
      }; break;
      default: {
        sprintf(operations_order[cost - 1], "sub %s -> %s\n", bytes_h, bytes_r);
        operation_stack[SUBSTITUTION_IDX] += 1;
      };
    }
    row = (row > -1) ? row - 1 : row;
    col = (col > -1) ? col - 1 : col;
    cost--;
  }
  return operations_order;
}

void init(const struct DistanceMatrix *d) {
  d->matrix[0] = PyUnicode_Compare(PyList_GetItem(d->ref, 0), PyList_GetItem(d->hyp, 0)) ? 1 : 0;
}

void incr_row(const struct DistanceMatrix *d, int idx, int cost) {
  d->matrix[idx] = d->matrix[idx - 1] + cost;
}

void incr_col(const struct DistanceMatrix *d, int idx, int cost) {
  d->matrix[idx * d->hyp_size] = d->matrix[(idx - 1) * d->hyp_size] + cost;
}

int get_state(const struct DistanceMatrix *d, int row, int col) {
  return (row == 0 && col == 0)
    ? STATE_FIRST_CELL : (row == 0)
    ? STATE_FIRST_ROW : (col == 0)
    ? STATE_FIRST_COL : STATE_BOUNEDED;
}

void prn_matrix(const struct DistanceMatrix *d) {
  int row = 0;
  int col = 0;
  printf("\n");
  for (row = 0; row < d->ref_size; row++) {
    for (col = 0; col < d->hyp_size; col++) {
      printf(" %d|", d->matrix[(row * d->hyp_size) + col]);
    }
    printf("\n");
  }
}
// ================================================================================


static PyObject* levenshtein_error_matrix(int cost, int* operations, char** operation_order) {
  /**
   * Returns a PyList [(iiii), List[str]] -> [(cost, deletion, insertion, substitution), ["steps"]]
   */
  PyObject* list = PyList_New(2);
  PyObject* error_tuple = Py_BuildValue("(iiii)",
                       cost,
                       operations[DELETION_IDX],
                       operations[INSERTION_IDX],
                       operations[SUBSTITUTION_IDX]);
  PyObject* error_detail = PyList_New(cost);

  PyList_SetItem(list, 0, error_tuple);
  for(int i = 0; i < cost; i++) {
    PyList_SetItem(error_detail, i, Py_BuildValue("s", operation_order[i]));
  }
  PyList_SetItem(list, 1, error_detail);
  free(operations);
  free(operation_order);
  return list;
}

static PyObject* levenshtein_edit_distance(PyObject* ref, PyObject* hyp) {
  int ref_size = PyList_Size(ref);
  int hyp_size = PyList_Size(hyp);
  char** empty_ref_operation_order = calloc(1, sizeof(char*));
  char** empty_hyp_operation_order = calloc(1, sizeof(char*));

  printf("ref_size: %d", ref_size);
  printf("hyp_size: %d", hyp_size);

  if (!hyp_size) {
    empty_hyp_operation_order[0] = (char*)calloc(100, sizeof(char));
    DEFAULT_OPERATIONS[INSERTION_IDX] = ref_size;
    sprintf(empty_hyp_operation_order[0], "Insert all chars in reference");
    return levenshtein_error_matrix(fmax(ref_size, hyp_size),
                                    DEFAULT_OPERATIONS,
                                    empty_hyp_operation_order);
  } else if (!ref_size) {
    empty_ref_operation_order[0] = (char*)calloc(100, sizeof(char));
    sprintf(empty_ref_operation_order[0], "Delete all chars in hypothesis");
    DEFAULT_OPERATIONS[DELETION_IDX] = hyp_size;
    return levenshtein_error_matrix(fmax(ref_size, hyp_size),
                                    DEFAULT_OPERATIONS,
                                    empty_ref_operation_order);
  }

  int* distances = (int*)calloc(ref_size * hyp_size, sizeof(int));
  int* DEFAULT_OPERATIONS = (int*) calloc(3, sizeof(int));
  DEFAULT_OPERATIONS[0] = 0;
  DEFAULT_OPERATIONS[1] = 0;
  DEFAULT_OPERATIONS[2] = 0;

  struct DistanceMatrix d = {
                             distances,
                             DEFAULT_OPERATIONS,
                             ref,
                             hyp,
                             ref_size,
                             hyp_size,

                             cell_above,
                             cell_before,
                             cell_diag_above,
                             get_final_cost,
                             get_state,
                             get_operation_order,

                             str_match,
                             get_operations_to_cell,

                             optimize,
                             incr_row,
                             incr_col,
                             init,
                             prn_matrix
  };

  int state = 0;

  for (int i = 0; i < ref_size; i++) {
    for (int j = 0; j < hyp_size; j++) {
      state = d.get_state(&d, i, j);
      switch (state) {
        case 0: d.init(&d); break;
        case 1: d.incr_row(&d, j, XFM_EFFORT); break;
        case 2: d.incr_col(&d, i, XFM_EFFORT); break;
        case 3: d.optimize(&d, i, j); break;
      }
    }
  }

  int transform_cost = d.get_final_cost(&d);
  int* operation_stack = (int*) calloc(OPERATIONS, sizeof(int));
  char** operation_order = d.get_operation_order(&d, transform_cost, operation_stack);

  free(distances);
  return levenshtein_error_matrix(transform_cost, operation_stack, operation_order);
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
