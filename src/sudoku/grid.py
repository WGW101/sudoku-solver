from os import PathLike
from typing import Iterator, ClassVar
from numpy.typing import ArrayLike
from numpy.random import Generator

import logging
import numpy as np


logger = logging.getLogger(__name__)


class SudokuGrid:
    @classmethod
    def from_txt_file(cls, path: PathLike) -> "SudokuGrid":
        with open(path) as f:
            txt_data = f.read()
        try:
            return cls.from_raw_str(txt_data)
        except ValueError:
            return cls.from_pretty_str(txt_data)

    @classmethod
    def from_raw_str(cls, raw_str: str) -> "SudokuGrid":
        return cls(
            [
                [0 if v == "." else int(v) for v in raw_str[9 * i : 9 * (i + 1)]]
                for i in range(9)
            ]
        )

    @classmethod
    def from_pretty_str(cls, pretty_str: str) -> "SudokuGrid":
        return cls(
            [
                [0 if v == "." else int(v) for v in l.split() if v != "|"]
                for l in pretty_str.splitlines()
                if not l.startswith("-")
            ]
        )

    @classmethod
    def generate(cls, seed: int | None = None) -> "SudokuGrid":
        if seed is not None:
            SudokuGrid._rng = np.random.default_rng(seed)
        raise NotImplementedError("WIP")

    _rng: ClassVar[Generator] = np.random.default_rng()

    _grid: np.ndarray
    _valid: np.ndarray

    def __init__(self, grid: ArrayLike | None = None) -> None:
        self._grid = np.zeros((9, 9), np.uint8) if grid is None else np.asarray(grid)
        self._valid = np.empty((10, 9, 9), np.bool_)
        self._reset_valid()

    def __str__(self) -> str:
        return "\n----- + ----- + -----\n".join(
            "\n".join(
                " | ".join(
                    " ".join(str(cell) if cell > 0 else "." for cell in block_cols)
                    for block_cols in row
                )
                for row in blocks_row
            )
            for blocks_row in self._grid.reshape(3, 3, 3, 3)
        )

    def is_valid(self) -> bool:
        return np.all((self._grid > 0) | (self._valid[1:].sum(0) > 0)).item()

    def save(self, path: PathLike | str, force: bool = False) -> bool:
        try:
            with open(path, "wt" if force else "xt") as f:
                f.write(str(self))
            return True
        except OSError as e:
            logger.error(e)
            return False

    def write(self, v: int | str, i: int, j: int, *, unsafe: bool = True) -> bool:
        if isinstance(v, str):
            v = 0 if v == "." else int(v)
        if unsafe and self._valid[0, i, j] or self._valid[v, i, j]:
            self._grid[i, j] = v
            self._update_valid(v, i, j)
            return True
        else:
            return False

    def erase(self, i: int, j: int) -> bool:
        return self.write(0, i, j)

    def clear(self) -> None:
        self._grid[self._valid[0]] = 0
        self._reset_valid()

    def copy(self) -> "SudokuGrid":
        cp = object.__new__(type(self))
        cp._grid = self._grid.copy()
        cp._valid = self._valid.copy()
        return cp

    def solve(self) -> bool:
        try:
            self._loop_uniq_valid_choices()
        except ValueError:
            return False
        if self._is_solved():
            return True
        for v, i, j in self._branch_less_vals():
            cp = self.copy()
            cp.write(v, i, j, unsafe=False)
            if cp.solve():
                self._grid = cp._grid
                self._valid = cp._valid
                return True
        else:
            return False

    def hint(self, level: int = 1) -> str:
        hints = [(0, v, i, j) for v, i, j in self._iter_uniq_valid_vals()]
        hints.extend((1, v, i, j) for v, i, j in self._iter_uniq_valid_rows())
        hints.extend((2, v, i, j) for v, i, j in self._iter_uniq_valid_cols())
        hints.extend((3, v, i, j) for v, i, j in self._iter_uniq_valid_blk_cells())
        if not hints:
            return "No basic hint available!"
        kind, v, i, j = SudokuGrid._rng.choice(hints)
        if level < 1:
            if kind == 0:
                return "A cell can only contain one value."
            elif kind == 1:
                return "A value in a column can only go in one row."
            elif kind == 2:
                return "A value in a row can only go in one column."
            else:
                return "A value in a block can only go in a cell."
        elif level == 1:
            if kind == 0:
                return f"A cell can only contain value {v}."
            elif kind == 1:
                return f"A value in a column can only go in row {i}."
            elif kind == 2:
                return f"A value in a row can only go in column {j}."
            else:
                cell = i % 3, j % 3
                return f"A value in a block can only go in cell {cell}."
        elif level == 2:
            if kind == 0:
                blk = i // 3, j // 3
                return f"A cell in block {blk} can only contain one value."
            elif kind == 1:
                return f"A value in column {j} can only go in one row."
            elif kind == 2:
                return f"A value in row {i} can only go in one column."
            else:
                blk = i // 3, j // 3
                return f"A value in block {blk} can only go in one cell."
        elif level == 3:
            if kind == 0:
                return f"Cell ({i}, {j}) can only contain one value."
            elif kind == 1:
                return f"Value {v} in column {j} can only go in one row."
            elif kind == 2:
                return f"Value {v} in row {i} can only go in one column."
            else:
                blk = i // 3, j // 3
                return f"Value {v} in block {blk} can only go in one cell."
        else:
            if kind == 0:
                return f"Cell ({i}, {j}) can only contain value {v}."
            elif kind == 1:
                return f"Value {v} in column {j} can only go in row {i}."
            elif kind == 2:
                return f"Value {v} in row {i} can only go in column {j}."
            else:
                blk, cell = zip(divmod(i, 3), divmod(j, 3))
                return f"Value {v} in block {blk} can only go in cell {cell}."

    def possible_values(self, i: int, j: int) -> set[int]:
        vv = np.flatnonzero(self._valid[1:, i, j]) + 1
        return {int(v) for v in vv}

    def _reset_valid(self) -> None:
        self._valid.fill(True)
        blk_view = self._grid.reshape(3, 3, 3, 3)
        blk_ii, ii, blk_jj, jj = np.nonzero(blk_view)
        vv = blk_view[blk_ii, ii, blk_jj, jj]
        self._valid[vv, blk_ii * 3 + ii, :] = False
        self._valid[vv, :, blk_jj * 3 + jj] = False
        self._valid[:, blk_ii * 3 + ii, blk_jj * 3 + jj] = False
        self._valid.reshape(10, 3, 3, 3, 3)[vv, blk_ii, :, blk_jj, :] = False

    def _update_valid(self, v: int, i: int, j: int) -> None:
        if v == 0:
            return
        self._valid[v, i, :] = False
        self._valid[v, :, j] = False
        self._valid[1:, i, j] = False
        blk_i = i // 3
        blk_j = j // 3
        self._valid.reshape(10, 3, 3, 3, 3)[v, blk_i, :, blk_j, :] = False

    def _iter_uniq_valid_vals(self) -> Iterator[tuple[int, int, int]]:
        ii, jj = np.nonzero(self._valid[1:].sum(0) == 1)
        vv, perm = np.nonzero(self._valid[1:, ii, jj])
        vv += 1
        yield from zip(vv, ii[perm], jj[perm])

    def _iter_uniq_valid_rows(self) -> Iterator[tuple[int, int, int]]:
        vv, jj = np.nonzero(self._valid[1:].sum(1) == 1)
        vv += 1
        perm, ii = np.nonzero(self._valid[vv, :, jj])
        yield from zip(vv[perm], ii, jj[perm])

    def _iter_uniq_valid_cols(self) -> Iterator[tuple[int, int, int]]:
        vv, ii = np.nonzero(self._valid[1:].sum(2) == 1)
        vv += 1
        perm, jj = np.nonzero(self._valid[vv, ii, :])
        yield from zip(vv[perm], ii[perm], jj)

    def _iter_uniq_valid_blk_cells(self) -> Iterator[tuple[int, int, int]]:
        blk_view = self._valid.reshape(10, 3, 3, 3, 3)
        vv, blk_ii, blk_jj = np.nonzero(blk_view[1:].sum((2, 4)) == 1)
        vv += 1
        perm, ii, jj = np.nonzero(blk_view[vv, blk_ii, :, blk_jj, :])
        yield from zip(vv[perm], blk_ii[perm] * 3 + ii, blk_jj[perm] * 3 + jj)

    def _iter_uniq_valid_choices(self) -> Iterator[tuple[int, int, int]]:
        yield from self._iter_uniq_valid_vals()
        yield from self._iter_uniq_valid_rows()
        yield from self._iter_uniq_valid_cols()
        yield from self._iter_uniq_valid_blk_cells()

    def _write_uniq_valid_choices(self) -> int:
        w_cnt = 0
        for v, i, j in self._iter_uniq_valid_choices():
            if self.write(v, i, j, unsafe=False):
                w_cnt += 1
            else:
                raise ValueError(f"Unexpected invalid write {v} at ({i}, {j})")
        return w_cnt

    def _loop_uniq_valid_choices(self) -> int:
        w_tot = 0
        while w_cnt := self._write_uniq_valid_choices():
            w_tot += w_cnt
        return w_tot

    def _branch_less_vals(self) -> Iterator[tuple[int, int, int]]:
        ii, jj = np.nonzero(self._grid == 0)
        # cnts = self._valid[:, ii, jj].sum(0)
        # whr = np.flatnonzero(cnts == cnts.min())
        whr = self._valid[:, ii, jj].sum(0).argmin()
        # TODO: Better var selection: min domain, max domain, rand?
        # ii, jj = ii[whr], jj[whr]
        i, j = ii[whr], jj[whr]
        vv = np.flatnonzero(self._valid[1:, i, j]) + 1
        # vv, perm = np.nonzero(self._valid[1:, ii, jj])
        # vv += 1
        # yield from zip(vv, ii[perm], jj[perm])
        yield from ((v, i, j) for v in vv)

    def _is_solved(self) -> bool:
        return np.all(self._grid > 0).item()
