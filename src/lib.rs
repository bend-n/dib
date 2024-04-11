//! # [DIB, also known as BMP](https://en.wikipedia.org/wiki/BMP_file_format) is a image format for bitmap storage.
//!
//! this crate implements [encoding](encode) and [decoding](decode) most types of BMPs.
#![allow(incomplete_features, mixed_script_confusables, internal_features)]
#![warn(missing_docs)]
#![feature(
    maybe_uninit_uninit_array_transpose,
    maybe_uninit_write_slice,
    generic_const_exprs,
    array_try_from_fn,
    slice_as_chunks,
    core_intrinsics,
    slice_flatten,
    effects,
    test
)]
mod until;

mod decode;
mod encode;

pub use decode::{decode, decode_into};
pub use encode::{encode, size as encoded_size};

pub use Color::*;
#[derive(Copy, Debug, Clone, PartialEq, Eq)]
#[repr(u8)]
/// Color types.
pub enum Color {
    /// Grayscale
    Y = 1,
    /// Grayscale with alpha
    YA,
    /// Red, green, blue
    RGB,
    /// RGB with alpha
    RGBA,
}

impl Color {
    /// Bits per pixel ([`depth`](Color::depth) [*](std::ops::Mul) [`8`](https://en.wikipedia.org/wiki/8)).
    #[must_use]
    pub const fn bpp(self) -> u8 {
        self.depth() * 8
    }

    /// Color depth.
    #[must_use]
    pub const fn depth(self) -> u8 {
        self as u8
    }
}

#[test]
fn encode_decode() {
    for elem in std::fs::read_dir("tdata").unwrap().map(Result::unwrap) {
        if elem.path().extension().unwrap() == "imgbuf" {
            let (dat, c, dim) =
                decode(&mut std::fs::File::open(elem.path().with_extension("bmp")).unwrap())
                    .unwrap();
            let mut v = vec![];
            encode(c, dim, &dat, &mut v).unwrap();
            assert_eq!(decode(&mut &v[..]).unwrap().0, dat);
        }
    }
}
