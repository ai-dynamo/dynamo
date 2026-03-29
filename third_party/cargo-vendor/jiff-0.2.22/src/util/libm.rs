/*!
A no-std module for some floating point operations.

These were vendored from the [`libm`] crate. For no-std specifically, I
wouldn't necessarily mind depending on `libm`, but I don't think there's a
way to express "only depend on a crate if some on-by-default feature is not
enabled."

We also very intentionally have very minimal floating point requirements. It's
used for `Span::total` where it's part of the API and thus necessary. It's also
used in floating point conversion routings to `jiff::SignedDuration`.

[`libm`]: https://github.com/rust-lang/libm
*/

pub(crate) trait Float {
    fn round(self) -> Self;
    fn trunc(self) -> Self;
    fn fract(self) -> Self;
}

const TOINT64: f64 = 1. / f64::EPSILON;

impl Float for f64 {
    fn round(self) -> f64 {
        (self + copysign64(0.5 - 0.25 * f64::EPSILON, self)).trunc()
    }

    fn trunc(self) -> f64 {
        let x = self;
        let x1p120 = f64::from_bits(0x4770000000000000); // 0x1p120f === 2 ^ 120

        let mut i: u64 = x.to_bits();
        let mut e: i64 = (i >> 52 & 0x7ff) as i64 - 0x3ff + 12;
        let m: u64;

        if e >= 52 + 12 {
            return x;
        }
        if e < 12 {
            e = 1;
        }
        m = -1i64 as u64 >> e;
        if (i & m) == 0 {
            return x;
        }
        core::hint::black_box(x + x1p120);
        i &= !m;
        f64::from_bits(i)
    }

    fn fract(self) -> f64 {
        self - self.trunc()
    }
}

impl Float for f32 {
    fn round(self) -> f32 {
        (self + copysign32(0.5 - 0.25 * f32::EPSILON, self)).trunc()
    }

    fn trunc(self) -> f32 {
        let x = self;
        let x1p120 = f32::from_bits(0x7b800000); // 0x1p120f === 2 ^ 120

        let mut i: u32 = x.to_bits();
        let mut e: i32 = (i >> 23 & 0xff) as i32 - 0x7f + 9;
        let m: u32;

        if e >= 23 + 9 {
            return x;
        }
        if e < 9 {
            e = 1;
        }
        m = -1i32 as u32 >> e;
        if (i & m) == 0 {
            return x;
        }
        core::hint::black_box(x + x1p120);
        i &= !m;
        f32::from_bits(i)
    }

    fn fract(self) -> f32 {
        self - self.trunc()
    }
}

fn copysign64(x: f64, y: f64) -> f64 {
    let mut ux = x.to_bits();
    let uy = y.to_bits();
    ux &= (!0) >> 1;
    ux |= uy & (1 << 63);
    f64::from_bits(ux)
}

fn copysign32(x: f32, y: f32) -> f32 {
    let mut ux = x.to_bits();
    let uy = y.to_bits();
    ux &= 0x7fffffff;
    ux |= uy & 0x80000000;
    f32::from_bits(ux)
}
