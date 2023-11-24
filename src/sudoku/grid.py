from os import PathLike
from typing import Iterator
from numpy.typing import ArrayLike
from numpy.random import Generator

import numpy as np


class SudokuGrid:
    @classmethod
    def read_txt(cls, path: PathLike) -> "SudokuGrid":
        with open(path) as f:
            txt_data = f.read()
        return cls(
            [
                [0 if v == "." else int(v) for v in l.split() if v != "|"]
                for l in txt_data.splitlines()
                if not l.startswith("-")
            ]
        )

    @classmethod
    def generate(
        cls, seed: int | None = None, *, rng: Generator | None = None
    ) -> "SudokuGrid":
        if rng is None or seed is not None:
            rng = np.random.default_rng(seed)
        raise NotImplementedError("WIP")

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
        vv, perm = np.nonzero(self._valid[:, ii, jj])
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
