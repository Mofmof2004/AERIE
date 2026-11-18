# Rust notes

Questions I asked while building AERIE, and mistakes the compiler caught. Raw material
for a cheat sheet.

## Questions

### How do I break out of an outer loop?

Give the outer loop a label (`'name:`) and break to it.

```rust
'search: for i in 0..n {
    for j in 0..n {
        if found(i, j) {
            break 'search; // leaves both loops
        }
    }
}
```

`continue 'search;` works the same way: it goes to the next round of the outer loop.

### How do I pull from a queue?

`VecDeque` from `std::collections`: push at the back, pop at the front (first in,
first out). `pop_front` returns an `Option`, `None` when the queue is empty.

```rust
use std::collections::VecDeque;

let mut queue: VecDeque<usize> = VecDeque::new();
queue.push_back(3);
while let Some(i) = queue.pop_front() {
    // runs until the queue is empty
}
```

`while let` repeats as long as the pattern matches: the standard BFS loop.

### How do I represent a matrix? (faer)

```rust
use faer::Mat;

let mut a: Mat<f64> = Mat::zeros(n, m); // n rows, m columns, all 0.0
let i3: Mat<f64> = Mat::identity(3, 3);
a[(i, j)] = 2.5;                         // index with a tuple (row, col)
let x = a[(i, j)];
let built = Mat::from_fn(n, m, |i, j| (i + j) as f64); // one call per cell
```

### How do I multiply two matrices? (faer)

Use the operators on references, so neither matrix is used up.

```rust
let c = &a * &b;       // product
let d = &a - &b;       // difference
let e = 0.5 * &a;      // scale by a number
let f = &a * &b + &c;  // they chain
```

`a * b` without `&` moves `a` and `b`: you can't use them afterwards.

### How do I get n (the size of a matrix)?

```rust
let n = p.nrows(); // number of rows
let m = p.ncols(); // number of columns
let k = labels.len(); // for a Vec or a slice
```

### How do I solve M X = B? (faer)

```rust
use faer::linalg::solvers::Solve; // the trait must be in scope

let x = m.partial_piv_lu().solve(&b); // LU factorization, never an inverse
```

### What do `iter()` and `flatten()` do?

`iter()` hands out a collection's items one at a time, as references (`&T`).
`flatten()` unpacks items that are themselves containers. For `Option`s: `Some(c)`
gives `c`, `None` is skipped.

```rust
let labels = vec![Some(1), None, Some(0)];
for &c in labels.iter().flatten() {
    // c = 1, then 0
}
// same as: for i in 0..labels.len() { if let Some(c) = labels[i] { ... } }
```

`&c` in the pattern takes the value out of the reference, so `c` is a plain `usize`.

## Mistakes the compiler caught

| Mistake | Fix | Why |
|---|---|---|
| `1 / d` with `d: f64` | `1.0 / d` | `1` is an integer; Rust never mixes integer and float silently. |
| `y[i, c]` | `y[(i, c)]` | faer indexes with one tuple, not two arguments. |
| `labels[i] == 0` with `Option<usize>` | `labels[i] == Some(0)`, or `if let Some(c) = labels[i]` | An `Option` is not a number. Class 0 is a real label, `None` means no label. |
| `x = -p` when you meant to subtract | `x -= p` | `= -p` overwrites; `-= p` subtracts from what is there. |
| `let y = (a = b);` | `a = b; let y = a;` | An assignment evaluates to `()`, not to the value. |
| `a.mul(b)` | `&a * &b` | `.mul` needs the `Mul` trait in scope; the operator is clearer. |
| `(1.0 - alpha) * y_0` inside a loop | `(1.0 - alpha) * &y_0` | Without `&`, `y_0` is moved in round 1 and gone in round 2. |
| `no method named solve` | `use faer::linalg::solvers::Solve;` | A trait's methods only exist where the trait is imported. |
| `let mut` on a variable never changed | drop the `mut` | Warning only, but it tells the reader the value is fixed. |
| Flag set to `true` once, never reset | `let mut flag = false;` **inside** the loop | Otherwise it stays `true` from an earlier round. |
