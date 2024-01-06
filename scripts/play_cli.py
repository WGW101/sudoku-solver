from argparse import ArgumentParser, Namespace
from pathlib import Path
import textwrap
from sudoku.grid import SudokuGrid


def parse_args() -> Namespace:
    parser = ArgumentParser()
    src_grp = parser.add_mutually_exclusive_group()
    src_grp.add_argument("--raw", "-r")
    src_grp.add_argument("--file", "-f", type=Path, default=Path("data/saved-grid.txt"))
    return parser.parse_args()


def main(args: Namespace) -> None:
    grid = (
        SudokuGrid.from_txt_file(args.file)
        if args.raw is None
        else SudokuGrid.from_raw_str(args.raw)
    )
    while True:
        print(grid)
        usr_in = input("> ")
        match list(usr_in):
            case ["x", *_]:
                break
            case ["s", *_]:
                grid.save("data/saved-grid.txt", force=True)
            case ["w", v, i, j, *_]:
                grid.write(*(int(c) for c in (v, i, j)))
            case ["e", i, j, *_]:
                grid.erase(*(int(c) for c in (i, j)))
            case ["c", *_]:
                grid.clear()
            case ["h", l, *_]:
                print(grid.hint(int(l)))
            case ["p", i, j, *_]:
                print(grid.possible_values(*(int(c) for c in (i, j))))
            case ["f", *_]:
                grid.solve()
            case ["v", *_]:
                print(f"Grid is {'valid.' if grid.is_valid() else 'INVALID!'}")
            case ["?", *_]:
                print(
                    textwrap.dedent(
                        """\
                        Valid inputs:
                        - x: Exit
                        - s: Save progress
                        - w<vij>: Write v at i,j
                        - e<ij>: Erase cell at i,j
                        - c: Clear grid
                        - h<l>: Get hint at level l
                        - p<ij>: Show possible vals at i,j
                        - f: Fill with solution
                        - v: Check grid validity
                        - ?: Show this usage"""
                    )
                )
            case _:
                print("Unrecognized input, ignoring")


if __name__ == "__main__":
    main(parse_args())
