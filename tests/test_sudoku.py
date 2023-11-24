from sudoku import SudokuGrid

if __name__ == "__main__":
    with open("data/example-grid.txt") as f:
        expect_0 = f.read()
    g = SudokuGrid.read_txt("data/example-grid.txt")
    assert str(g) == expect_0
    print(expect_0)
    print(g._write_uniq_valid_choices())
    print(g._write_uniq_valid_choices())
    print(g._write_uniq_valid_choices())
    expect_3 = str(g)
    print(expect_3)

    g.clear()
    assert str(g) == expect_0
    g._loop_uniq_valid_choices()
    assert str(g) == expect_3

    cp = g.copy()
    assert str(cp) == expect_3

    g.solve()
    print(g)
