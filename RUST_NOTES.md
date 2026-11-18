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

### What does `max_by_key(...)` with `Reverse` and `.filter` do?

```rust
let winner = (0..n_classes)
    .max_by_key(|&c| (counts[c], Reverse(c)))
    .filter(|&c| counts[c] > 0);
```

- `(0..n_classes)`: an iterator over the class numbers.
- `max_by_key(|&c| key)`: returns the item whose key is largest, as an `Option` (`None` if
  there are no items). On a tie it returns the **last** maximum.
- Tuples compare left to right: first the count, then, on a tie, the second element.
- `Reverse(c)` (from `std::cmp`) flips the comparison: `Reverse(1) > Reverse(2)`. So on a
  tie the **smallest** class wins.
- `Option::filter(cond)`: keeps `Some(c)` if `cond` holds, otherwise gives `None`. Here: no
  winner when nobody reported.

Same thing as a loop:

```rust
let mut winner = None;
let mut best = 0;
for c in 0..n_classes {
    if counts[c] > best { best = counts[c]; winner = Some(c); } // strict >: ties keep the smaller class
}
```

### What does `Reverse` do?

`std::cmp::Reverse` wraps a value and flips how it compares: `Reverse(1) > Reverse(2)`. The
value itself is unchanged (`Reverse(5).0 == 5`). Used inside a key to flip one direction:

```rust
v.sort_by_key(|&x| Reverse(x));          // biggest first
max_by_key(|&c| (counts[c], Reverse(c))) // biggest count, then smallest c on a tie
```

### What does `iter().map(...).collect()` do? And `Option::or`?

Take a list, transform each item, gather the results into a new list.

```rust
let guesses: Vec<Option<usize>> = labels.iter().map(|&l| l.or(winner)).collect();
```

- `.map(|&l| ...)`: a closure (inline function) applied to every item; what it returns is
  the new item. `&l` takes the value out of the reference.
- `l.or(winner)`: `l` if it is `Some`, otherwise `winner`.
- `.collect()`: builds the collection named in the type annotation (`Vec<...>` here); the
  annotation is required, or `collect` doesn't know what to build.

Same as:

```rust
let mut guesses = Vec::new();
for &l in labels.iter() {
    guesses.push(l.or(winner));
}
```

### What does `assert_eq!(a, b, "message")` do?

Checks `a == b`. If equal, nothing happens; if not, the program panics (stops) and prints
both values and the message. Used at the top of a function to fail early, with a clear
message, when the inputs don't fit what the function assumes.

```rust
assert_eq!(labels.len(), n, "one label slot per drone");
assert!(alpha < 1.0, "alpha must be below 1"); // any condition
assert_ne!(n, 0);                              // not equal
debug_assert!(x >= 0.0);                       // checked in debug builds only
```

Tests use the same macros: a test passes if nothing panics.

### What does `(0..n).filter(...).collect()` do?

Keeps the **indices** that pass a test, and gathers them into a collection.

```rust
// labels = [Some(0), None, None, Some(1), None]  ->  queue = [0, 3]
let mut queue: VecDeque<usize> = (0..n).filter(|&i| labels[i].is_some()).collect();
```

- `(0..n)`: the drone numbers. `.filter(|&i| cond)`: keeps `i` only if `cond` is `true`.
- It keeps the numbers `i`, not `labels[i]`. For the values themselves:
  `labels.iter().flatten()`.
- `map` transforms every item (same count out); `filter` keeps or drops items unchanged
  (maybe fewer out). `collect` can build a `Vec`, a `VecDeque`, a `HashSet`… whatever the
  annotation says.

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
