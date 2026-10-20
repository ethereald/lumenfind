use std::path::Path;

use anyhow::{Context, Result, bail};
use image::{DynamicImage, ImageBuffer, ImageReader, Rgba};

pub fn load_image(path: &Path) -> Result<DynamicImage> {
    if is_heif(path) {
        return load_heif(path);
    }
    ImageReader::open(path)
        .with_context(|| format!("could not open {}", path.display()))?
        .with_guessed_format()?
        .decode()
        .with_context(|| format!("could not decode {}", path.display()))
}

fn is_heif(path: &Path) -> bool {
    matches!(
        path.extension()
            .and_then(|extension| extension.to_str())
            .map(str::to_ascii_lowercase)
            .as_deref(),
        Some("heic" | "heif")
    )
}

#[cfg(windows)]
fn load_heif(path: &Path) -> Result<DynamicImage> {
    decode_heif_with_wic(path).with_context(|| {
        format!(
            "could not decode HEIC/HEIF photo {}; install Microsoft's HEIF Image Extensions (and HEVC Video Extensions if prompted)",
            path.display()
        )
    })
}

#[cfg(not(windows))]
fn load_heif(path: &Path) -> Result<DynamicImage> {
    bail!(
        "HEIC/HEIF decoding requires Windows Imaging Component: {}",
        path.display()
    )
}

#[cfg(windows)]
fn decode_heif_with_wic(path: &Path) -> Result<DynamicImage> {
    use std::{os::windows::ffi::OsStrExt, ptr};

    use windows::{
        Win32::{
            Foundation::{GENERIC_READ, RPC_E_CHANGED_MODE},
            Graphics::Imaging::{
                CLSID_WICImagingFactory, GUID_WICPixelFormat32bppRGBA, IWICImagingFactory,
                IWICPalette, WICBitmapDitherTypeNone, WICBitmapPaletteTypeCustom,
                WICDecodeMetadataCacheOnDemand,
            },
            System::Com::{
                CLSCTX_INPROC_SERVER, COINIT_MULTITHREADED, CoCreateInstance, CoInitializeEx,
                CoUninitialize,
            },
        },
        core::PCWSTR,
    };

    struct ComApartment(bool);

    impl ComApartment {
        fn initialize() -> Result<Self> {
            let result = unsafe { CoInitializeEx(None, COINIT_MULTITHREADED) };
            if result == RPC_E_CHANGED_MODE {
                return Ok(Self(false));
            }
            result.ok().context("could not initialize Windows COM")?;
            Ok(Self(true))
        }
    }

    impl Drop for ComApartment {
        fn drop(&mut self) {
            if self.0 {
                unsafe { CoUninitialize() };
            }
        }
    }

    let _apartment = ComApartment::initialize()?;
    let factory: IWICImagingFactory =
        unsafe { CoCreateInstance(&CLSID_WICImagingFactory, None, CLSCTX_INPROC_SERVER) }
            .context("could not create the Windows Imaging Component factory")?;
    let wide_path = path
        .as_os_str()
        .encode_wide()
        .chain(Some(0))
        .collect::<Vec<_>>();
    let decoder = unsafe {
        factory.CreateDecoderFromFilename(
            PCWSTR(wide_path.as_ptr()),
            None,
            GENERIC_READ,
            WICDecodeMetadataCacheOnDemand,
        )
    }
    .context("Windows Imaging Component could not open this HEIC/HEIF file")?;
    let frame = unsafe { decoder.GetFrame(0) }.context("HEIC/HEIF file has no image frame")?;
    let converter = unsafe { factory.CreateFormatConverter() }
        .context("could not create a Windows image format converter")?;
    unsafe {
        converter.Initialize(
            &frame,
            &GUID_WICPixelFormat32bppRGBA,
            WICBitmapDitherTypeNone,
            None::<&IWICPalette>,
            0.0,
            WICBitmapPaletteTypeCustom,
        )
    }
    .context("could not convert HEIC/HEIF image pixels to RGBA")?;

    let mut width = 0_u32;
    let mut height = 0_u32;
    unsafe { converter.GetSize(&mut width, &mut height) }
        .context("could not read HEIC/HEIF image dimensions")?;
    let stride = width
        .checked_mul(4)
        .context("HEIC/HEIF image dimensions are too large")?;
    let buffer_len = usize::try_from(stride)
        .ok()
        .and_then(|stride| {
            usize::try_from(height)
                .ok()
                .and_then(|height| stride.checked_mul(height))
        })
        .context("HEIC/HEIF image dimensions are too large")?;
    if width == 0 || height == 0 {
        bail!("HEIC/HEIF image has empty dimensions");
    }
    let mut pixels = vec![0_u8; buffer_len];
    unsafe { converter.CopyPixels(ptr::null(), stride, &mut pixels) }
        .context("could not decode HEIC/HEIF image pixels")?;
    let image = ImageBuffer::<Rgba<u8>, _>::from_raw(width, height, pixels)
        .context("Windows returned an invalid HEIC/HEIF pixel buffer")?;
    Ok(DynamicImage::ImageRgba8(image))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_heif_extensions_case_insensitively() {
        assert!(is_heif(Path::new("IMG_8854.HEIC")));
        assert!(is_heif(Path::new("photo.heif")));
        assert!(!is_heif(Path::new("photo.jpg")));
    }
}
