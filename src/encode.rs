use std::io::{self, Write};

use super::Color;
use super::Color::*;
use atools::prelude::*;
use bites::*;
use raad::le::*;

/// uses the so-called `BITMAPINFOHEADER`
const DIB_HEADER_SIZE: u32 = 40;
/// `BITMAPV4HEADER`
const DIB_HEADER_V4_SIZE: u32 = 108;
const CORE_HEADER_SIZE: u32 = 14;

/// Size of encoded dib.
pub fn size(color: Color, (width, height): (u32, u32)) -> u32 {
    pal(color) * 4 + dat_size(color, (width, height))
}

fn pal(color: Color) -> u32 {
    (color.depth() < 3).then_some(256).unwrap_or(0)
}

fn dat_size(color: Color, (width, height): (u32, u32)) -> u32 {
    width * height * color.depth() as u32 + height * ((4 - (width * color.depth() as u32) % 4) % 4)
}

fn dib_hdr_v4(color: Color, (width, height): (u32, u32)) -> [u8; DIB_HEADER_V4_SIZE as _] {
    le(DIB_HEADER_V4_SIZE)
        .couple(le(width))
        .couple(le(height))
        // planes??
        .couple(le::<u16>(1))
        .couple(le(color.bpp() as u16))
        // bitfield compression
        .couple(le::<u32>(3))
        .couple(le(dat_size(color, (width, height))))
        // "pixels per metre"
        .couple(le::<u32>(0))
        .couple(le::<u32>(0))
        // color count
        .couple(le::<u32>(pal(color)))
        .couple(le::<u32>(0))
        // bitfields
        .couple(le::<u32>(0x00ff0000))
        .couple(le::<u32>(0x0000ff00))
        .couple(le::<u32>(0x000000ff))
        .couple(le::<u32>(0xff000000))
        .couple(*b"sRGB")
        // endpoints
        .couple([le::<u32>(0); 3 * 3].flatten())
        // gamma
        .couple([le::<u32>(0); 3].flatten())
}

fn dib_hdr(color: Color, (width, height): (u32, u32)) -> [u8; DIB_HEADER_SIZE as _] {
    le(DIB_HEADER_SIZE)
        .couple(le(width))
        .couple(le(height))
        // planes??
        .couple(le::<u16>(1))
        .couple(le(color.bpp() as u16))
        // compression method (only interesting for RLE grayscale) (cant use due to support issues)
        .couple(le::<u32>(0))
        .couple(le(dat_size(color, (width, height))))
        // "pixels per metre"
        .couple(le::<u32>(0))
        .couple(le::<u32>(0))
        // color count
        .couple(le::<u32>(pal(color)))
        .couple(le::<u32>(0))
}

fn hdr(color: Color, dib: u32, (width, height): (u32, u32)) -> [u8; 14] {
    b"BM"
        // fs
        .couple(le(dib + CORE_HEADER_SIZE + size(color, (width, height))))
        // "reserved 1"
        .couple([0; 2])
        // "reserved 2"
        .couple([0; 2])
        // file offset (length of previous bytes) (who designed this format) (why is this necessary)
        .couple(le(dib + CORE_HEADER_SIZE + pal(color) * 4))
}

/// Encode a BMP/DIB.

/// # Panics
///
/// if your width * height * color depth isnt data's length
pub fn encode(
    color: Color,
    (width, height): (u32, u32),
    data: impl AsRef<[u8]>,
    to: &mut impl Write,
) -> io::Result<()> {
    let data = data.as_ref();
    assert_eq!(
        (width as usize * height as usize)
            .checked_mul(color.depth() as usize)
            .unwrap(),
        data.len(),
        "please dont lie to me"
    );

    unsafe fn rgba(width: u32, data: &[u8], to: &mut impl Write) -> io::Result<()> {
        data.as_chunks_unchecked::<4>()
            .chunks_exact(width as _)
            .map(|x| x.iter().map(|&[r, g, b, a]| [b, g, r, a]))
            .rev()
            .flatten()
            .try_for_each(|x| to.w(x))?;
        Ok(())
    }

    unsafe fn rgb(width: u32, data: &[u8], to: &mut impl Write) -> io::Result<()> {
        data.as_chunks_unchecked::<3>()
            .chunks_exact(width as _)
            .map(|x| x.iter().map(|&[r, g, b]| [b, g, r]))
            .rev()
            .try_for_each(|mut x| {
                x.try_for_each(|x| to.w(x))?;
                to.w(&[0; 4][..width as usize % 4])
            })?;

        Ok(())
    }

    const GRAY: [u8; 256 * 4] = car::map!(range::<256>(), |x| [x as u8; 3].join(0)).flatten();
    unsafe fn ya(width: u32, data: &[u8], to: &mut impl Write) -> io::Result<()> {
        to.w(GRAY)?;
        data.as_chunks_unchecked::<2>()
            .chunks_exact(width as _)
            .map(|x| x.iter().map(|&[x, _]| x))
            .rev()
            .try_for_each(|mut x| {
                x.try_for_each(|x| to.w(x))?;
                to.w(&[0; 4][..width as usize % 4])
            })?;

        Ok(())
    }

    fn y(width: u32, data: &[u8], to: &mut impl Write) -> io::Result<()> {
        to.w(GRAY)?;
        data.chunks_exact(width as _).rev().try_for_each(|row| {
            to.w(row)?;
            to.w(&[0; 4][..width as usize % 4])
        })?;
        Ok(())
    }

    unsafe {
        match color {
            Y | YA | RGB => to.w(hdr(color, DIB_HEADER_SIZE, (width, height))
                .couple(dib_hdr(color, (width, height))))?,
            RGBA => to.w(hdr(color, DIB_HEADER_V4_SIZE, (width, height))
                .couple(dib_hdr_v4(color, (width, height))))?,
        }
        match color {
            Y => y(width, data, to),
            YA => ya(width, data, to),
            RGB => rgb(width, data, to),
            RGBA => rgba(width, data, to),
        }
    }
}
