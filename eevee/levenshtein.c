#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <math.h>

const int OPERATIONS = 3;

// Read as transformation cost
const int XFM_EFFORT = 1;
const int DELETION_IDX = 0;
const int INSERTION_IDX = (DELETION_IDX + 1);
const int SUBSTITUTION_IDX = (INSERTION_IDX + 1);
const int STATE_FIRST_CELL = 0;
const int STATE_FIRST_ROW = (STATE_FIRST_CELL + 1);
const int STATE_FIRST_COL = (STATE_FIRST_ROW + 1);
const int STATE_BOUNEDED = (STATE_FIRST_COL + 1);
int LIST_OPERATIONS[] = {0, 0, 0};
int* DEFAULT_OPERATIONS = LIST_OPERATIONS;
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
  int* (*get_best_operations)(const struct DistanceMatrix*);

  // side-effects:
  void (*optimize)(const struct DistanceMatrix*, int row, int col);
  void (*incr_row)(const struct DistanceMatrix*, int idx, int cost);
  void (*incr_col)(const struct DistanceMatrix*, int idx, int cost);
  void (*init)(const struct DistanceMatrix*);
  void (*prn_matrix)(const struct DistanceMatrix*);
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
  /**
   * Calculates the transformation effort needed to convert
   * the reference-substring to hypothesis-substring.
   * Cost of transformation is `XFM_EFFORT` for deletion
   * and insertion errors.
   *
   * The effort for substitution seems a bit different but
   * it is just for the special case where the characters match.
   * In the match-case we don't need to transform, hence the value
   * obtained from `chars_match` 0/1 is the effort needed too.
   */
  int left_cell_cost = cell_before(d, row, col) + XFM_EFFORT;
  int upper_cell_cost = cell_above(d, row, col) + XFM_EFFORT;
  int chars_match = PyUnicode_Compare(PyList_GetItem(d->ref, row), PyList_GetItem(d->hyp, col)) ? 1 : 0;
  int diag_cell_cost = cell_diag_above(d, row, col) + chars_match;
  printf("chars match :: %d ||", chars_match);
  d->matrix[row * d->hyp_size + col] = fmin(left_cell_cost, fmin(upper_cell_cost, diag_cell_cost));
}

int get_final_cost(const struct DistanceMatrix *d) {
  return d->matrix[d->hyp_size * d->ref_size - 1];
}

int* get_best_operations(const struct DistanceMatrix *d) {
  /**
   * The objective of this function is to return an array(size=3; for each operation)
   * of integers in which, each index contains:
   * 0 - Deletion cost
   * 1 - Insertion cost
   * 2 - Substitution cost
   */
  int deletion_idx = (d->hyp_size - 1) * d->ref_size - 1;
  int insertion_idx = d->hyp_size * (d->ref_size - 1) - 1;
  int substitution_idx = (d->hyp_size - 1) * (d->ref_size - 1) - 1;
  int* operations = calloc(3, sizeof(int));
  operations[0] = d->matrix[deletion_idx];
  operations[1] = d->matrix[insertion_idx];
  operations[2] = d->matrix[substitution_idx];

  int penultimate_operation = d->get_final_cost(d) - XFM_EFFORT;
  int i = 0;
  for(i = 0; i < OPERATIONS; i++) {
    operations[i] = penultimate_operation != operations[i] ? 0 : operations[i];

    /**
     * This section adds the final action on the basis of the row and the
     * size of hypothesis vs reference.
     *
     * - If the length of hypothesis > reference, the last action
     * would be a deletion.
     * - If the length of hypothesis < reference, the last action
     * would be an insertion.
     * - If the lengths are same, the last action would be a substitution.
     */
    operations[i] = (d->ref_size > d->hyp_size && i == DELETION_IDX) ? operations[i] + 1 : operations[i];
    operations[i] = (d->ref_size < d->hyp_size && i == INSERTION_IDX) ? operations[i] + 1 : operations[i];
    operations[i] = (d->ref_size == d->hyp_size && i == SUBSTITUTION_IDX) ? operations[i] + 1 : operations[i];
  }
  return operations;
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
      printf(" %d |", d->matrix[(row * d->hyp_size) + col]);
    }
    printf("\n");
  }
}
// ================================================================================


static PyObject* levenshtein_error_matrix(int cost, int *best_operation) {
  /**
   * Returns a PyDict {char*: long}
   * The keys correspond to cost and, operations; Contain integer values: 0 if operation was costlier
   * but > 0 if operation had minimum cost.
   * "cost": cost of transforming the `reference` string to the `hypothesis` string.
   */
  PyObject *error_matrix;
  error_matrix = PyDict_New();
  PyDict_SetItemString(error_matrix, STR_COST, PyLong_FromLong(cost));
  PyDict_SetItemString(error_matrix, STR_DELETION, PyLong_FromLong(best_operation[DELETION_IDX]));
  PyDict_SetItemString(error_matrix, STR_INSERTION, PyLong_FromLong(best_operation[INSERTION_IDX]));
  PyDict_SetItemString(error_matrix, STR_SUBSTITUTION, PyLong_FromLong(best_operation[SUBSTITUTION_IDX]));
  return error_matrix;
}

static PyObject* levenshtein_edit_distance(PyObject* ref, PyObject* hyp) {
  int ref_size = PyList_Size(ref);
  int hyp_size = PyList_Size(hyp);

  if (!fmin(ref_size, hyp_size)) {
    return levenshtein_error_matrix(fmax(ref_size, hyp_size), DEFAULT_OPERATIONS);
  }

  int* distances = (int *)calloc(ref_size * hyp_size, sizeof(int));
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
                             get_best_operations,

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
        case STATE_FIRST_CELL: d.init(&d); break;
        case STATE_FIRST_ROW: d.incr_row(&d, j, XFM_EFFORT); break;
        case STATE_FIRST_COL: d.incr_col(&d, i, XFM_EFFORT); break;
        case STATE_BOUNEDED: d.optimize(&d, i, j); break;
      }
    }
  }

  int transform_cost = d.get_final_cost(&d);
  int* best_operation = d.get_best_operations(&d);
  free(distances);
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
