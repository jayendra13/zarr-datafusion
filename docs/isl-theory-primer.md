# isl / barvinok — build, FFI, and the Phase-8 linking spike

Ground truth from actually building `isl` + `barvinok` and calling them from Rust,
so Phase 8 (the `polyhedral` Tier-B backend) starts from facts, not guesses. Doubles
as the "primer" the [implementation plan](exact-cardinality-implementation-plan.md)
§8 references. Result up front:

> **Linking `isl` + `barvinok` from Rust and getting exact coupled-polytope counts
> works on glibc (proven). Static-musl — what the CLI release binary uses — does
> *not* work as-is and needs a full musl-cross rebuild of the whole chain, with
> NTL/C++ the crux.**

The spike lives outside the crate (built in a scratchpad, not committed); this doc is
the reproducible recipe + verdicts.

## The dependency chain (as it actually is)

```
barvinok  →  isl (bundled)  →  GMP        (all C)
         ↘  NTL (mandatory, C++)  →  libstdc++
```

**Correction to the original plan:** barvinok **0.41.9 requires NTL**. Its configure
option is `--with-ntl=system|build` (default `system`) — there is **no `--without-ntl`
"off" value**, so the pure-C build the plan assumed is not available on modern
barvinok. NTL is C++, which is why it pulls in `libstdc++` and makes the musl story
hard.

## Build recipe (verified on Ubuntu / glibc / x86_64)

Versions that worked: barvinok-0.41.9, bundled isl 0.23 (`libisl.so.23.5.0`),
GMP 6.3.0, NTL (system), autoconf 2.72 / automake 1.18.1 / libtool 2.5.4.

```bash
# 1. build + link deps (autotools, GMP dev, NTL dev, git)
sudo apt-get install -y \
    autoconf automake libtool libtool-bin \
    libgmp-dev libntl-dev build-essential git

# 2. source (canonical repo; pin a release, not HEAD)
git clone https://repo.or.cz/barvinok.git barvinok
cd barvinok
git checkout barvinok-0.41.9
git submodule update --init --recursive     # bundled isl (+ pet) — one isl version

# 3. bootstrap + configure
#    --with-isl=bundled  : use barvinok's own isl (no skew with system libisl.so.23)
#    --with-ntl=system   : NTL is mandatory (NOT --without-ntl)
#    CFLAGS=-fPIC         : needed to link the static .a into a Rust cdylib/binary
./autogen.sh
./configure --prefix="$PWD/../install" --with-isl=bundled --with-ntl=system CFLAGS="-fPIC"

# 4. build + install + sanity check
make -j"$(nproc)"
make install
echo 'card { [x,y] : 0 <= x <= y < 10 };' | ../install/bin/iscc     # -> { 55 }
```

Artifacts: `install/lib/{libbarvinok.a, libisl.a, libisl.so.*}`,
`install/include/{barvinok,isl}/*.h`, `install/bin/{iscc, barvinok_count, …}`.
Note `make` logs `CXXLD libbarvinok.la` — the C++ (NTL) link, confirmed.

## FFI recipe (Rust → barvinok), verified

A one-function C shim keeps the FFI surface tiny and dodges `isl-rs` version skew (it
`#include`s barvinok's *own* isl headers, so one consistent isl):

```c
/* csrc/barvinok_shim.c */
#include <isl/ctx.h>
#include <isl/set.h>
#include <isl/val.h>
#include <barvinok/isl.h>          /* isl_set_count_val (barvinok) */

long bv_count(const char *str) {   /* integer points of a BOUNDED set; <0 = error */
    isl_ctx *ctx = isl_ctx_alloc();
    if (!ctx) return -1;
    isl_set *set = isl_set_read_from_str(ctx, str);
    if (!set) { isl_ctx_free(ctx); return -2; }
    isl_val *v = isl_set_count_val(set);          /* keeps set */
    long n = v ? isl_val_get_num_si(v) : -3;
    isl_val_free(v); isl_set_free(set); isl_ctx_free(ctx);
    return n;
}
```

`build.rs` (via the `cc` crate) — the **link order that resolved cleanly**: barvinok
and isl **static**, gmp/ntl/stdc++ **dynamic**:

```rust
cc::Build::new().file("csrc/barvinok_shim.c").include("<install>/include").compile("barvinok_shim");
println!("cargo:rustc-link-search=native=<install>/lib");
println!("cargo:rustc-link-lib=static=barvinok");
println!("cargo:rustc-link-lib=static=isl");
println!("cargo:rustc-link-lib=dylib=gmp");
println!("cargo:rustc-link-lib=dylib=ntl");     // C++
println!("cargo:rustc-link-lib=dylib=stdc++");  // C++ runtime for NTL
```

Rust side: `extern "C" { fn bv_count(s: *const c_char) -> c_long; }` + a safe wrapper.
The passing tests (first-try compile, ~2s):

| set | count |
|---|---|
| `{ [x,y] : 0 <= x <= y < 10 }` (simplex) | **55** |
| `{ [x,y] : 0 <= x < 10 and 0 <= y < 10 }` (box) | **100** |
| `{ [i,j] : 0 <= i < 10 and i <= j < i + 3 }` (**coupled** band) | **30** |

The band is the point: a set Tier A can't represent, counted exactly over FFI.

**API note:** `isl_set_count_val(set) → isl_val` is the clean one-shot for a bounded,
parameter-free set; `isl_val_get_num_si` extracts the count (denominator is 1). The
general path (parametric sets) is `isl_set_card → isl_pw_qpolynomial`, evaluated at the
parameter point — needed only once we count *parametric* families.

## Verdicts

| Target | Verdict | Notes |
|---|---|---|
| **glibc / dynamic** | ✅ **proven** | build + link + FFI + exact coupled counts all work |
| **static-musl** (CLI release) | ❌ **not viable as-is** | two blockers below |

**Static-musl blockers (measured):**
1. **No musl C/C++ toolchain** — `musl-gcc` / `x86_64-linux-musl-gcc` / `musl-g++`
   absent; `cc-rs` can't even compile the shim for the musl target.
2. **The chain is glibc** — the built `libbarvinok.a`/`libisl.a` and system
   `libgmp`/`libntl` are glibc-ABI; they cannot link into a static-musl binary.

Reaching musl means **rebuilding gmp + isl + barvinok + NTL against musl** with a
cross-toolchain — and **cross-building C++ (NTL) + a C++ runtime for musl is the hard,
fragile part** (foreshadowed by NTL turning out mandatory).

## Recommendations / follow-ups

- **Ship `polyhedral` as a glibc-only, dynamically-linked feature.** The static-musl
  release binary simply omits it (feature off) — the `apply → None` fallback in the
  `IndexSet` abstraction already handles "no Tier B" cleanly, so nothing breaks.
- **If musl is ever required:** the highest-leverage move is *not* a musl-C++
  toolchain — it's **dropping NTL**. Check whether an **older barvinok** supports the
  NTL-free build (isl-internal arithmetic); removing C++ makes musl far more tractable.
  Pull that thread before committing to a musl-C++ cross build.
- **Productionizing into the crate** (beyond this spike): replace the hardcoded install
  path with a vendored source build or a `pkg-config`/system discovery in `build.rs`;
  gate everything behind the `polyhedral` cargo feature (default off, verified no-op);
  and — the prerequisite for any *plan* change on join queries — bridge the exact count
  into DataFusion `Statistics` (pushdown-admission and partition fan-out already consume
  our cardinality directly; join order/algorithm do not).

## Theory pointer

The counting itself is **Ehrhart / Barvinok** lattice-point enumeration: barvinok
computes the (piecewise, parametric) quasi-polynomial giving the number of integer
points of a polytope as a function of its parameters; for a fixed bounded polytope that
collapses to the constant `isl_set_count_val` returns. This is exactly why "product of
axis extents" is the wrong number for coupled regions — see
[`exact-cardinality-tier-b-use-cases.md`](exact-cardinality-tier-b-use-cases.md) for the
motivating queries and how to measure the q-error improvement.
