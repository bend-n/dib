use crate::until::Until;
use crate::Color;
use atools::prelude::*;
use raad::le::*;
use std::intrinsics::unlikely;
use std::io::{self, Read};
use std::iter::{repeat, repeat_with};
use std::mem::MaybeUninit as MU;
use std::{error, fmt};

pub const CORE_HEADER_SIZE: u32 = 12;
pub const DIB_HEADER_SIZE: u32 = 40;
pub const DIB_HEADER_V2_SIZE: u32 = 52;
pub const DIB_HEADER_V3_SIZE: u32 = 56;
pub const DIB_HEADER_V4_SIZE: u32 = 108;
pub const DIB_HEADER_V5_SIZE: u32 = 124;

/// Decoding errors.
#[derive(Debug)]
pub enum Error {
    /// `BM` signature wrong (not `BM`)
    NoSig,
    /// Run length encoded data seems invalid
    BadRLE,
    /// Image type unsupported/invalid
    BadImageType(u32),
    /// Header size not one of {12, 40, 52, 56, 108, 124}
    BadDibHdrSize(u32),
    /// width/height == 0
    SizeZero,
    /// width * height > 2^32
    SizeLarge,
    /// Number of bpp insensible.
    BadBitsPerPixel(u8),
    /// IO error occurred while decoding image
    Io(io::Error),
    /// Allocator error occurred while decoding image
    Allocation(Box<dyn std::any::Any + Send + 'static>),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Allocation(_) => f.write_str("allocation failure"),
            Error::NoSig => f.write_str("not a BMP"),
            Error::BadRLE => f.write_str("Corrupt RLE data"),
            Error::BadDibHdrSize(x) => write!(
                f,
                "Header size ({x}) not one of {{12, 40, 52, 56, 108, 124}}"
            ),
            Error::SizeLarge => f.write_str("size too large"),
            Error::SizeZero => f.write_str("size cannot be 0"),
            Error::BadBitsPerPixel(x) => write!(f, "{x} is nonsensical number of bits per pixel"),
            Error::BadImageType(ty) => match ty {
                4 => f.write_str("JPEG compression not supported (consider using plain jpeg)"),
                5 => f.write_str(
                    "PNG compression not supported (consider using plain png (pronounced ping)",
                ),
                11..=13 => f.write_str("CMYK format not supported (not a printer)"),
                _ => write!(f, "unknown image type {ty:x}"),
            },
            Error::Io(x) => write!(f, "{x}"),
        }
    }
}

impl From<std::io::Error> for Error {
    fn from(value: std::io::Error) -> Self {
        Self::Io(value)
    }
}

impl From<Box<dyn std::any::Any + 'static + Send>> for Error {
    fn from(value: Box<dyn std::any::Any + 'static + Send>) -> Self {
        Self::Allocation(value)
    }
}

impl error::Error for Error {
    fn source(&self) -> Option<&(dyn error::Error + 'static)> {
        match self {
            Self::Io(x) => Some(x),
            _ => None,
        }
    }
}
pub type Result<T> = std::result::Result<T, Error>;

#[derive(PartialEq, Eq, Clone, Copy, Debug)]
struct Field {
    shift: u8,
    // 0..=8
    len: u8,
}

// 0b11000000 => (6, 2)
fn field(mask: u32) -> Option<Field> {
    if mask == 0 {
        Some(Field { shift: 0, len: 1 })
    } else {
        let mut shift = mask.trailing_zeros() as u8;
        let mut len = mask.count_ones() as u8;
        // fat mask
        if len > 8 {
            shift += len - 8;
            len = 8;
        }
        if unlikely(len == 0) {
            return None;
        }
        Some(Field { shift, len })
    }
}

impl Field {
    fn read(self, data: u32) -> u8 {
        let data = data >> self.shift;
        let data = data & ((1 << self.len) - 1);
        // precise:
        // return (((data & ((1 << self.len) - 1)) as f32 / ((1 << self.len) - 1) as f32)
        //     * 0xff as f32) as u8;
        match self.len {
            1 => (data * 0xff) as u8,
            2 => (data * 0x55) as u8,
            // these arent quite precise
            3 => cvtn::<3>(data),
            4 => ((data) << 4 | (data)) as u8,
            5 => cvtn::<5>(data),
            6 => cvtn::<6>(data),
            7 => (((data >> 6) & 1) + (2 * data)) as u8,
            8 => data as u8,
            x => unreachable!("{x}"),
        }
    }
}

fn cvtn<const TO: u32>(x: u32) -> u8 {
    let max_src = (1 << TO) - 1;
    let m = (255 << 24) / max_src + 1;
    let x = (x) as u8;
    ((x as u32 * m) >> 24) as u8
}

#[derive(PartialEq, Eq, Copy, Clone, Debug)]
struct Fields {
    r: Field,
    g: Field,
    b: Field,
    a: Field,
}

impl Fields {
    fn new(r_mask: u32, g_mask: u32, b_mask: u32, a_mask: u32) -> Option<Self> {
        Some(Self {
            r: field(r_mask)?,
            g: field(g_mask)?,
            b: field(b_mask)?,
            a: field(a_mask)?,
        })
    }
}

/// returns: offset
pub fn sig(r: &mut impl Read) -> Result<u32> {
    let Ok(b"BM") = r.r::<[u8; 2]>().as_ref() else {
        return Err(Error::NoSig);
    };
    // 00 00 00 00 (fs), 00 (r1), 00
    r.r::<[u8; 8]>()?;
    Ok(r.r::<u32>()?) // file offset
}

fn rem_fs(dib: u32) -> u32 {
    (dib - 40).saturating_sub(match dib {
        DIB_HEADER_SIZE => 4 * 3,
        _ => 4 * 4,
    })
}

fn fields(rd: &mut impl Read, dib: u32) -> io::Result<Fields> {
    let [r, g, b] = std::array::try_from_fn(|_| rd.r::<u32>())?;
    let a = match dib {
        DIB_HEADER_SIZE => 0,
        _ => rd.r::<u32>()?,
    };
    Fields::new(r, g, b, a).ok_or(std::io::Error::new(
        std::io::ErrorKind::InvalidData,
        "bad fields",
    ))
}

fn pal<const N: usize>(r: &mut impl Read, pal: usize) -> Result<Vec<[u8; 3]>> {
    let mut all = (0..pal).map(|_| r.r::<[u8; N]>().map(|x| [x[2], x[1], x[0]]));
    let ret = all.by_ref().take(256).collect::<io::Result<Vec<_>>>()?;
    all.for_each(drop);
    Ok(ret)
}

macro_rules! rev {
    ($if:ident, $with:expr, $($then:tt)+) => {
        if $if {
            $with $($then)+
        } else {
            $with.rev() $($then)+
        }
    };
}

unsafe fn palletted<const BITS: usize, const PAL: usize>(
    r: &mut impl Read,
    w: u32,
    buf: &mut [MU<u8>],
    split: impl FnMut(u8) -> [u8; 8 / BITS] + Copy,
    pal_size: usize,
    rev: bool,
) -> Result<()> {
    let pal = pal::<PAL>(r, pal_size)?;
    rev![
        rev,
        buf.as_chunks_unchecked_mut::<3>().chunks_exact_mut(w as _),
        .try_for_each(|row| {
            std::iter::repeat_with(|| r.r::<u8>())
                .take(((BITS as u32 * w + 31) / 32 * 4) as usize)
                .until_done(|mut input_| {
                    row.iter_mut()
                        .zip(
                            input_
                                .by_ref()
                                .flat_map(split)
                                .map(|x| *pal.get(x as usize).unwrap_or(&[0; 3])),
                        )
                        .for_each(|(px, x)| {
                            MU::copy_from_slice(px, &x);
                        });
                    input_.for_each(drop);
                })
        })?
    ];
    Ok(())
}

unsafe fn one_two_four_eight<const PAL: usize>(
    r: &mut impl Read,
    bpp: u8,
    w: u32,
    buf: &mut [MU<u8>],
    pal_size: usize,
    rev: bool,
) -> Result<()> {
    match bpp {
        1 => palletted::<1, PAL>(
            r,
            w,
            buf,
            |b| {
                atools::range::<8>()
                    .rev()
                    .map(|x| b & (1 << x) != 0)
                    .map(u8::from)
            },
            pal_size,
            rev,
        ),
        2 => palletted::<2, PAL>(
            r,
            w,
            buf,
            |b| atools::range::<4>().rev().map(|x| b >> (2 * x) & 0b11),
            pal_size,
            rev,
        ),
        4 => palletted::<4, PAL>(r, w, buf, |b| [b >> 4, b & 0x0f], pal_size, rev),
        8 => palletted::<8, PAL>(r, w, buf, |b| [b], pal_size, rev),
        _ => unreachable!(),
    }
}

unsafe fn sixteen(r: &mut impl Read, buf: &mut [MU<u8>], w: u32, rev: bool) -> Result<()> {
    let pad = w as usize % 2 * 2;
    let pad = &mut [0; 2][..pad];
    rev![rev, buf.as_chunks_unchecked_mut::<3>()
        .chunks_exact_mut(w as _),
        .try_for_each(|px| {
            px.iter_mut()
                .try_for_each(|px| {
                    r.r::<u16>().map(|x| {
                        [
                            field(0b11111 << 10).unwrap().read(x as _),
                            field(0b11111 << 5).unwrap().read(x as _),
                            field(0b11111).unwrap().read(x as _)
                        ]
                    }).map(|x| *px = MU::new(x).transpose())
                })
                .and_then(|()| r.read_exact(pad))
        })?];
    Ok(())
}

unsafe fn twenty_four(r: &mut impl Read, buf: &mut [MU<u8>], w: u32, rev: bool) -> Result<()> {
    let pad = &mut [0; 4][..(4 - (w as usize * 3) % 4) % 4];
    rev![rev, buf.as_chunks_unchecked_mut::<3>()
        .chunks_exact_mut(w as _),
        .try_for_each(|px| {
            px.iter_mut().try_for_each(|px| {
                r.r::<[u8; 3]>().map(|[b, g, r]| {
                    *px = MU::new([r, g, b]).transpose();
                })})
            .and_then(|()| r.read_exact(pad))
    })?];
    Ok(())
}

unsafe fn thirty_two(r: &mut impl Read, buf: &mut [MU<u8>], w: u32, rev: bool) -> Result<()> {
    rev![rev, buf.as_chunks_unchecked_mut::<3>()
        .chunks_exact_mut(w as _),
        .flatten()
        .try_for_each(|px| {
            r.r::<[u8; 4]>().map(|[b, g, r, _]| {
                *px = MU::new([r, g, b]).transpose();
            })
        })?];

    Ok(())
}

unsafe fn rle(
    r: &mut impl Read,
    buf: &mut [MU<u8>],
    pal_size: usize,
    w: u32,
    rev: bool,
    is_4: bool,
) -> Result<()> {
    let pal = pal::<4>(r, pal_size)?;
    let mut rows: Box<dyn Iterator<Item = &mut [[MU<u8>; 3]]>> = if rev {
        Box::new(buf.as_chunks_unchecked_mut::<3>().chunks_exact_mut(w as _))
    } else {
        Box::new(
            buf.as_chunks_unchecked_mut::<3>()
                .chunks_exact_mut(w as _)
                .rev(),
        )
    };

    while let Some(ro) = rows.next() {
        let mut p = ro.iter_mut();
        let mut x = 0;
        loop {
            let op = r.r::<u8>()?;
            let dat = r.r::<u8>()?;
            match op {
                0 if dat == 0 => {
                    break p.for_each(|x| *x = MU::new([0; 3]).transpose());
                } // end of row
                0 if dat == 1 => {
                    p.for_each(|x| *x = MU::new([0; 3]).transpose());
                    rows.for_each(|x| x.fill(MU::new([0; 3]).transpose()));
                    return Ok(());
                } // end of file
                0 if dat == 2 => {
                    // delta
                    let [xδ, yδ] = r.r::<[u8; 2]>()?;
                    if yδ > 0 {
                        p.by_ref().for_each(|x| *x = MU::new([0; 3]).transpose());
                        for _ in 1..yδ {
                            rows.next()
                                .ok_or(Error::BadRLE)?
                                .flatten_mut()
                                .fill(MU::new(0));
                        }
                        p = rows.next().ok_or(Error::BadRLE)?.iter_mut();
                        for _ in 0..x {
                            p.next().ok_or(Error::BadRLE)?.fill(MU::new(0));
                        }
                    }
                    for _ in 0..xδ {
                        p.next().ok_or(Error::BadRLE)?.fill(MU::new(0));
                    }
                    x += xδ as usize;
                }
                0 => {
                    // absolute
                    let dat = dat as usize;
                    if is_4 {
                        repeat_with(|| r.r::<u8>()).until_done(|x| {
                            x.flat_map(|x| [x >> 4, x & 0b0000_1111])
                                .map(|x| pal.get(x as usize).unwrap_or(&[0; 3]))
                                .take(dat)
                                .zip(p.by_ref())
                                .for_each(|(&x, y)| *y = MU::new(x).transpose());
                        })?;
                        if (dat + 1) / 2 % 2 == 1 {
                            r.r::<u8>()?;
                        }
                    } else {
                        for _ in 0..dat {
                            MU::copy_from_slice(
                                p.next().ok_or(Error::BadRLE)?,
                                pal.get(r.r::<u8>()? as usize).unwrap_or(&[0; 3]),
                            );
                        }
                        if dat % 2 == 1 {
                            r.r::<u8>()?;
                        }
                    }

                    x += dat;
                }
                n => {
                    // actual run length part (n, color)
                    if is_4 {
                        let col = [
                            pal.get(dat as usize >> 4).unwrap_or(&[0; 3]),
                            pal.get(dat as usize & 0b0000_1111).unwrap_or(&[0; 3]),
                        ];
                        p.by_ref()
                            .take(n as _)
                            .zip(repeat(col).flatten())
                            .for_each(|(a, b)| {
                                MU::copy_from_slice(a, b);
                            });
                    } else {
                        let c = pal.get(dat as usize).unwrap_or(&[0; 3]);
                        p.by_ref()
                            .take(n as _)
                            .for_each(|x| *x = MU::new(*c).transpose());
                    }
                    x += n as usize;
                }
            };
        }
    }
    Ok(())
}

/// Decodes a [DIB/BMP](https://en.wikipedia.org/wiki/BMP_file_format).
pub fn decode(r: &mut impl Read) -> Result<(Vec<u8>, Color, (u32, u32))> {
    let mut len = 0;
    let mut out = vec![];
    let (c, d) = unsafe {
        decode_into(r, |x| {
            len = x;
            out.try_reserve_exact(x).map_err(|x| {
                let x: Box<dyn std::any::Any + 'static + Send> = Box::new(x);
                x
            })?;
            Ok(out.spare_capacity_mut().as_mut_ptr())
        })?
    };
    unsafe { out.set_len(len) };
    Ok((out, c, d))
}

/// Decodes a [DIB/BMP](https://en.wikipedia.org/wiki/BMP_file_format).
/// Takes a function that will allocate `n` bytes, and return a pointer to the allocation.
///
/// # Safety
///
/// undefined behaviour if the allocation is not `n` bytes
///
/// ideally this would be `unsafe impl FnOnce`, but that doesnt work.
pub unsafe fn decode_into(
    r: &mut impl Read,
    buf: impl FnOnce(usize) -> std::result::Result<*mut MU<u8>, Box<dyn std::any::Any + 'static + Send>>,
) -> Result<(Color, (u32, u32))> {
    sig(r)?;
    let dib_size = r.r::<u32>()?;
    let mut rev = false;
    let (w, h) = match dib_size {
        CORE_HEADER_SIZE => (r.r::<u16>()? as u32, r.r::<u16>()? as u32), // 8
        DIB_HEADER_SIZE | DIB_HEADER_V2_SIZE | DIB_HEADER_V3_SIZE | DIB_HEADER_V4_SIZE
        | DIB_HEADER_V5_SIZE => {
            let w = r.r::<i32>()?.unsigned_abs(); // 8
            let h = r.r::<i32>()?; // 12
            rev = h < 0;
            (w, h.unsigned_abs())
        }
        x => return Err(Error::BadDibHdrSize(x)),
    };
    if w == 0 || h == 0 {
        return Err(Error::SizeZero);
    }
    if w.checked_mul(h).ok_or(Error::SizeLarge)? > 0xffff && cfg!(fuzzing) {
        return Err(Error::SizeLarge);
    };
    match dib_size {
        CORE_HEADER_SIZE => {
            let _planes = r.r::<u16>()? == 1; // 10
            let bpp = r.r::<u16>()? as u8; // 12

            match bpp {
                1 | 4 | 8 => {
                    let buf = buf(w as usize * h as usize * 3)?;
                    let buf = std::slice::from_raw_parts_mut(buf, w as usize * h as usize * 3);
                    one_two_four_eight::<3>(r, bpp, w, buf, 1 << bpp, false)?;
                    Ok((Color::RGB, (w, h)))
                }
                24 => {
                    let buf = buf(w as usize * h as usize * 3)?;
                    let buf = std::slice::from_raw_parts_mut(buf, w as usize * h as usize * 3);
                    twenty_four(r, buf, w, false)?;
                    Ok((Color::RGB, (w, h)))
                }
                x => Err(Error::BadBitsPerPixel(x)),
            }
        }
        DIB_HEADER_SIZE | DIB_HEADER_V2_SIZE | DIB_HEADER_V3_SIZE | DIB_HEADER_V4_SIZE
        | DIB_HEADER_V5_SIZE => {
            // handle overflow
            let _planes = r.r::<u16>()? == 1; // 14
            let bpp = r.r::<u16>()? as u8; // 16
            let compress = r.r::<u32>()?; // 20
            const RGB: u32 = 0;
            const RLE8: u32 = 1;
            const RLE4: u32 = 2;
            const BITFIELDS: u32 = 3;
            r.r::<[u8; 12]>()?; // 32
            let colors = match r.r::<u32>()? as u64 {
                0 => 1u64
                    .checked_shl(bpp as _)
                    .ok_or(Error::BadBitsPerPixel(bpp))?,
                n => n,
            }; // 36
            r.r::<u32>()?; // 40 (important colors)

            // this is the end of the dib_header_v1
            // if we are a cool dib header we may have things like bitfields

            match compress {
                RGB => {
                    (0..dib_size - 40).try_for_each(|_| r.r::<u8>().map(|_| ()))?;
                    let buf = buf(w as usize * h as usize * 3)?;
                    let buf = std::slice::from_raw_parts_mut(buf, w as usize * h as usize * 3);
                    match bpp {
                        1 | 2 | 4 | 8 => one_two_four_eight::<4>(r, bpp, w, buf, colors as _, rev)?,
                        16 => sixteen(r, buf, w, rev)?,
                        24 => twenty_four(r, buf, w, rev)?,
                        32 => thirty_two(r, buf, w, rev)?,
                        x => return Err(Error::BadBitsPerPixel(x)),
                    }
                    Ok((Color::RGB, (w, h)))
                }
                x @ (RLE4 | RLE8) => {
                    // https://learn.microsoft.com/en-us/windows/win32/gdi/bitmap-compression
                    let buf = buf(w as usize * h as usize * 3)?;
                    let buf = std::slice::from_raw_parts_mut(buf, w as usize * h as usize * 3);
                    rle(r, buf, colors as _, w, rev, x == RLE4)?;
                    Ok((Color::RGB, (w, h)))
                }
                BITFIELDS => match bpp {
                    16 => {
                        let b = fields(r, dib_size)?;
                        (0..rem_fs(dib_size)).try_for_each(|_| r.r::<u8>().map(|_| ()))?;
                        let pad = w as usize % 2 * 2;
                        let pad = &mut [0; 2][..pad];
                        let n = match b.a.len {
                            0 => w as usize * h as usize * 3,
                            _ => w as usize * h as usize * 4,
                        };
                        let buf = buf(n)?;
                        let buf = std::slice::from_raw_parts_mut(buf, n);
                        match b.a.len {
                            0 => rev![rev, buf.as_chunks_unchecked_mut::<3>()
                                    .chunks_exact_mut(w as _),
                                    .try_for_each(|px| {
                                        px.iter_mut()
                                            .try_for_each(|px| {
                                                r.r::<u16>().map(|x| {
                                                    [b.r.read(x as _), b.g.read(x as _), b.b.read(x as _)]                                            
                                                }).map(|x| *px = MU::new(x).transpose())
                                            })
                                            .and_then(|()| r.read_exact(pad))
                                    })?],
                            _ => rev![rev, buf.as_chunks_unchecked_mut::<4>()
                                .chunks_exact_mut(w as _),
                                .try_for_each(|px| {
                                    px.iter_mut()
                                        .try_for_each(|px| {
                                            r.r::<u16>().map(|x| {
                                                [b.r.read(x as _), b.g.read(x as _), b.b.read(x as _), b.a.read(x as _)]
                                            }).map(|x| *px = MU::new(x).transpose())
                                        })
                                        .and_then(|()| r.read_exact(pad))
                                })?],
                        }
                        Ok((
                            match b.a.len {
                                0 => Color::RGB,
                                _ => Color::RGBA,
                            },
                            (w, h),
                        ))
                    }
                    32 => {
                        let b = fields(r, dib_size)?;
                        (0..rem_fs(dib_size)).try_for_each(|_| r.r::<u8>().map(|_| ()))?;

                        let n = match b.a.len {
                            0 => w as usize * h as usize * 3,
                            _ => w as usize * h as usize * 4,
                        };
                        let buf = buf(n)?;
                        let buf = std::slice::from_raw_parts_mut(buf, n);
                        match b.a.len {
                            0 => rev![rev, buf.as_chunks_unchecked_mut::<3>()
                                    .chunks_exact_mut(w as _),
                                    .try_for_each(|px| {
                                        px.iter_mut()
                                            .try_for_each(|px| {
                                                r.r::<u32>().map(|x| {
                                                    [b.r.read(x), b.g.read(x), b.b.read(x)]
                                                }).map(|x| *px = MU::new(x).transpose())
                                            })
                                    })?],
                            _ => rev![rev, buf.as_chunks_unchecked_mut::<4>()
                                    .chunks_exact_mut(w as _),
                                    .try_for_each(|px| {
                                        px.iter_mut()
                                            .try_for_each(|px| {
                                                r.r::<u32>().map(|x| {
                                                    [b.r.read(x), b.g.read(x), b.b.read(x), b.a.read(x)]
                                                }).map(|x| *px = MU::new(x).transpose())
                                            })
                                    })?],
                        }
                        Ok((
                            match b.a.len {
                                0 => Color::RGB,
                                _ => Color::RGBA,
                            },
                            (w, h),
                        ))
                    }
                    x => Err(Error::BadBitsPerPixel(x)),
                },
                c => {
                    // could be "png compression" or something
                    Err(Error::BadImageType(c))
                }
            }
        }
        _ => unreachable!(),
    }
}
