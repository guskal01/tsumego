#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "structmember.h"
#include <random>
#include <tuple>

static constexpr int max_height = 10;
static constexpr int max_width = 10;
static constexpr int max_zobrist_entries = max_height*max_width*10;

static uint64_t zobrist_table[max_height][max_width][2];
static uint64_t zobrist_turn;
static int dr[] = {-1, 0, 1, 0};
static int dc[] = {0, 1, 0, -1};

// probably bad
static int visited_has_liberty[max_height][max_width];
static int visited_hyp_floodfill[max_height][max_width];

typedef struct {
    PyObject_HEAD
    int height;
    int width;
    int a[max_height][max_width];
    int turn;
    int passes;
    int moves;
    uint64_t zobrist_hash;
    int zobrist_seen_index;
    int zobrist_seen[max_zobrist_entries];
} BoardObject;

static void set_stone(BoardObject* self, int r, int c, int color){
    if(self->a[r][c] != 0) self->zobrist_hash ^= zobrist_table[r][c][self->a[r][c]-1];
    self->a[r][c] = color;
    if(self->a[r][c] != 0) self->zobrist_hash ^= zobrist_table[r][c][self->a[r][c]-1];
}

static void switch_turn(BoardObject* self){
    self->turn ^= 3;
    self->zobrist_hash ^= zobrist_turn;
}

static int is_inside(BoardObject* self, int r, int c){
    return r >= 0 && c >= 0 && r < self->height && c < self->width;
}

static void reset_superko(BoardObject* self){
    self->zobrist_seen[0] = self->zobrist_hash;
    self->zobrist_seen_index = 1;
}

static int zobrist_contains(BoardObject* self, int hash){
    for(int i = 0; i < self->zobrist_seen_index; i++){
        if(self->zobrist_seen[i] == hash) return 1;
    }
    return 0;
}

static int has_liberty(BoardObject* self, int r, int c){
    assert(is_inside(self, r, c));
    int col = self->a[r][c];

    // DFS
    memset(visited_has_liberty, 0, sizeof(visited_has_liberty));
    std::pair<int, int> stack[4*self->height*self->width];
    stack[0] = {r, c};
    int ind = 1;
    while(ind){
        std::tie(r, c) = stack[--ind];
        if(r < 0 || c < 0) continue;
        if(r >= self->height || c >= self->width){
            if(col == 1) return 1;
            continue;
        }
        if(visited_has_liberty[r][c]) continue;
        visited_has_liberty[r][c] = 1;
        if(self->a[r][c] == 0) return 1;
        if(self->a[r][c] != col) continue;
        for(int d = 0; d < 4; d++){
            stack[ind++] = {r+dr[d], c+dc[d]};
        }
    }
    return 0;
}

static int floodfill(BoardObject* self, int r, int c, int new_col){
    assert(is_inside(self, r, c));
    int old_col = self->a[r][c];

    // DFS
    std::pair<int, int> stack[4*self->height*self->width];
    stack[0] = {r, c};
    int ind = 1;
    while(ind){
        std::tie(r, c) = stack[--ind];
        if(!is_inside(self, r, c)) continue;
        if(self->a[r][c] != old_col) continue;
        set_stone(self, r, c, new_col);

        for(int d = 0; d < 4; d++){
            stack[ind++] = {r+dr[d], c+dc[d]};
        }
    }
    return 0;
}

static int hypothetical_floodfill(BoardObject* self, int r, int c){
    assert(is_inside(self, r, c));
    int old_col = self->a[r][c];

    // DFS
    std::pair<int, int> stack[4*self->height*self->width];
    stack[0] = {r, c};
    int ind = 1;
    uint64_t h = 0;
    while(ind){
        std::tie(r, c) = stack[--ind];
        if(!is_inside(self, r, c)) continue;
        if(self->a[r][c] != old_col) continue;
        if(visited_hyp_floodfill[r][c]) continue;
        visited_hyp_floodfill[r][c] = 1;
        assert(old_col != 0);
        h ^= zobrist_table[r][c][old_col-1];
        for(int d = 0; d < 4; d++){
            stack[ind++] = {r+dr[d], c+dc[d]};
        }
    }
    return h;
}

static int is_legal(BoardObject* self, int r, int c){
    if(r == -1 && c == -1) return 1; //pass
    if(!is_inside(self, r, c)) return 0;
    if(self->a[r][c] != 0) return 0;
    set_stone(self, r, c, self->turn);
    if(has_liberty(self, r, c)){
        if(zobrist_contains(self, self->zobrist_hash^zobrist_turn)){
            set_stone(self, r, c, 0);
            return 0; //uncommon but possible
        }
        set_stone(self, r, c, 0);
        return 1;
    }

    memset(visited_hyp_floodfill, 0, sizeof(visited_hyp_floodfill));
    int h = 0;
    for(int d = 0; d < 4; d++){
        if(is_inside(self, r+dr[d], c+dc[d]) && self->a[r+dr[d]][c+dc[d]] != self->turn && !has_liberty(self, r+dr[d], c+dc[d])){
            h ^= hypothetical_floodfill(self, r+dr[d], c+dc[d]);
        }
    }
    if(h == 0){
        set_stone(self, r, c, 0);
        return 0; // nothing captured, suicide
    }
    if(zobrist_contains(self, self->zobrist_hash^zobrist_turn^h)){
        set_stone(self, r, c, 0);
        return 0;
    }
    set_stone(self, r, c, 0);
    return 1;
}


static int Board_init(BoardObject* self, PyObject* args, PyObject* kwds){
    static char *kwlist[] = {(char*)"height", (char*)"width", NULL};
    if(!PyArg_ParseTupleAndKeywords(args, kwds, "ii", kwlist, &self->height, &self->width))
        return -1;
    self->turn = 1;
    return 0;
}

static PyObject* Board_set_stone(BoardObject* self, PyObject* args){
    int r, c, color;
    if(!PyArg_ParseTuple(args, "iii", &r, &c, &color)) return NULL;

    set_stone(self, r, c, color);

    Py_RETURN_NONE;
}

static PyObject* Board_get_stone(BoardObject* self, PyObject* args){
    int r, c;
    if(!PyArg_ParseTuple(args, "ii", &r, &c)) return NULL;

    return PyLong_FromLong(self->a[r][c]);
}

static PyObject* Board_switch_turn(BoardObject* self, PyObject* Py_UNUSED(ignored)){
    switch_turn(self);

    Py_RETURN_NONE;
}

static PyObject* Board_reset_superko(BoardObject* self, PyObject* Py_UNUSED(ignored)){
    reset_superko(self);

    Py_RETURN_NONE;
}

static PyObject* Board_is_legal(BoardObject* self, PyObject* args){
    int r, c;
    if(!PyArg_ParseTuple(args, "ii", &r, &c)) return NULL;

    if(is_legal(self, r, c)){
        Py_RETURN_TRUE;
    }
    Py_RETURN_FALSE;
}

static PyObject* Board_get_legal_moves(BoardObject* self, PyObject* Py_UNUSED(ignored)){
    PyObject* ret = PyList_New(0);
    PyList_Append(ret, Py_BuildValue("ii", -1, -1));
    for(int r = 0; r < self->height; r++){
        for(int c = 0; c < self->width; c++){
            if(is_legal(self, r, c)){
                PyList_Append(ret, Py_BuildValue("ii", r, c));
            }
        }
    }
    return ret;
}

static PyObject* Board_play(BoardObject* self, PyObject* args){
    int r, c;
    if(!PyArg_ParseTuple(args, "ii", &r, &c)) return NULL;

    assert(is_legal(self, r, c));
    if(r == -1 && c == -1){ //pass
        self->passes++;
        switch_turn(self);
        reset_superko(self);
        Py_RETURN_NONE;
    }
    self->passes = 0;
    set_stone(self, r, c, self->turn);
    for(int d = 0; d < 4; d++){
        if(is_inside(self, r+dr[d], c+dc[d]) && self->a[r+dr[d]][c+dc[d]] == (self->turn^3) && !has_liberty(self, r+dr[d], c+dc[d])){
            floodfill(self, r+dr[d], c+dc[d], 0);
        }
    }
    switch_turn(self);
    self->zobrist_seen[self->zobrist_seen_index++] = self->zobrist_hash;
    self->moves++;
    Py_RETURN_NONE;
}

static PyObject* Board_black_won(BoardObject* self, PyObject* Py_UNUSED(ignored)){
    for(int r = 0; r < self->height; r++){
        for(int c = 0; c < self->width; c++){
            if(self->a[r][c] == 2){
                Py_RETURN_FALSE;
            }
        }
    }
    Py_RETURN_TRUE;
}

static PyObject* Board_game_over(BoardObject* self, PyObject* Py_UNUSED(ignored)){
    if(self->passes >= 3 || self->moves >= self->height*self->width*5){
        Py_RETURN_TRUE;
    }
    Py_RETURN_FALSE;
}

static PyObject* Board_str(BoardObject* self){
    std::string s = "";
    for(int r = 0; r < self->height; r++){
        for(int c = 0; c < self->width; c++){
            s += ".BW"[self->a[r][c]];
            s += " ";
        }
        s += "B\n";
    }
    for(int c = 0; c < self->width; c++){
        s += "B ";
    }
    s += "\n";
    s += "?BW"[self->turn];
    s += " to play";
    return PyUnicode_FromString(s.c_str());
}

static PyMemberDef Board_members[] = {
    {"height", T_INT, offsetof(BoardObject, height), READONLY, "height"},
    {"width", T_INT, offsetof(BoardObject, width), READONLY, "width"},
    {"turn", T_INT, offsetof(BoardObject, turn), READONLY, "turn"},
    {"passes", T_INT, offsetof(BoardObject, passes), READONLY, "passes"},
    {"moves", T_INT, offsetof(BoardObject, moves), READONLY, "moves"},
    {"zobrist_hash", T_ULONGLONG, offsetof(BoardObject, zobrist_hash), READONLY, "zobrist hash of current situation"},
    {NULL}
};

static PyMethodDef Board_methods[] = {
    {"set_stone", (PyCFunction)Board_set_stone, METH_VARARGS, "Set color at intersection"},
    {"get_stone", (PyCFunction)Board_get_stone, METH_VARARGS, "Get color at intersection"},
    {"switch_turn", (PyCFunction)Board_switch_turn, METH_NOARGS, "Switches turn"},
    {"reset_superko", (PyCFunction)Board_reset_superko, METH_NOARGS, "Resets seen situations"},
    {"is_legal", (PyCFunction)Board_is_legal, METH_VARARGS, "Checks is move is legal"},
    {"get_legal_moves", (PyCFunction)Board_get_legal_moves, METH_NOARGS, "List of legal moves"},
    {"play", (PyCFunction)Board_play, METH_VARARGS, "Play move"},
    {"black_won", (PyCFunction)Board_black_won, METH_NOARGS, "Check if black won"},
    {"game_over", (PyCFunction)Board_game_over, METH_NOARGS, "Check if game is over (white won)"},
    {NULL}
};

static PyTypeObject BoardType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "tsumegoboard.Board",
    .tp_basicsize = sizeof(BoardObject),
    .tp_itemsize = 0,
    .tp_str = (reprfunc) Board_str,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_doc = "Tsumego board",
    .tp_methods = Board_methods,
    .tp_members = Board_members,
    .tp_init = (initproc) Board_init,
    .tp_new = PyType_GenericNew,
};

static PyModuleDef tsumegoboardmodule = {
    PyModuleDef_HEAD_INIT,
    .m_name = "tsumegoboard",
    .m_doc = "Tsumego board module",
    .m_size = -1,
};

PyMODINIT_FUNC
PyInit_tsumegoboard(void)
{
    // zobrist init
    std::random_device rd;
    std::mt19937_64 gen(rd());
    std::uniform_int_distribution<uint64_t> dis;
    for(int r = 0; r < max_height; r++){
        for(int c = 0; c < max_width; c++){
            for(int p = 0; p < 2; p++){
                zobrist_table[r][c][p] = dis(gen);
            }
        }
    }
    zobrist_turn = dis(gen);

    PyObject *m;
    if(PyType_Ready(&BoardType) < 0)
        return NULL;

    m = PyModule_Create(&tsumegoboardmodule);
    if(m == NULL)
        return NULL;

    Py_INCREF(&BoardType);
    if(PyModule_AddObject(m, "Board", (PyObject *) &BoardType) < 0) {
        Py_DECREF(&BoardType);
        Py_DECREF(m);
        return NULL;
    }

    return m;
}
