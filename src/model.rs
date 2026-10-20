use std::{
    fs::{self, File},
    io::{Read, Write},
    path::{Path, PathBuf},
    sync::OnceLock,
};

use anyhow::{Result, anyhow, bail};
use image::{DynamicImage, GenericImageView, imageops::FilterType};
use ndarray::{Array2, Array3};
use ort::{environment::Environment, session::Session, value::Tensor};
use sha2::{Digest, Sha256};
use tokenizers::Tokenizer;

use crate::db::EMBEDDING_DIM;
use crate::image_loader::load_image;

const REVISION: &str = "daa72c51243991dfcaf9f9137d2c573d8f7790c0";
const REPOSITORY: &str = "onnx-community/embeddinggemma-2-ONNX";
const PATCH_SIZE: usize = 16;
const POOLING: usize = 3;
const SOFT_TOKEN_BUDGET: usize = 70;
const MAX_PATCHES: usize = SOFT_TOKEN_BUDGET * POOLING * POOLING;
const IMAGE_TOKEN: i64 = 258_880;
const BEGIN_IMAGE_TOKEN: i64 = 255_999;
const END_IMAGE_TOKEN: i64 = 258_882;
const BOS_TOKEN: i64 = 2;

struct ModelFile {
    remote: &'static str,
    local: &'static str,
    sha256: &'static str,
}

const MODEL_FILES: &[ModelFile] = &[
    ModelFile {
        remote: "tokenizer.json",
        local: "tokenizer.json",
        sha256: "4d777ef5bdc1aa36227abdfb77c3e49e7b9c892d16e1b6bda41c393504828be4",
    },
    ModelFile {
        remote: "onnx/model_q4.onnx",
        local: "model_q4.onnx",
        sha256: "f9eeba97acddf139b8ee2ddf04bc30dceafa88de93fadf74d7644e0d61a477a9",
    },
    ModelFile {
        remote: "onnx/model_q4.onnx_data",
        local: "model_q4.onnx_data",
        sha256: "c3975f2d1ab7a1878ae31a7d7a9b7804a827aff3800b60dfceafce21cac3df49",
    },
    ModelFile {
        remote: "onnx/vision_encoder_q4.onnx",
        local: "vision_encoder_q4.onnx",
        sha256: "7ea284226d4938f0ad921ab091f1d80a9ca699aa802984ef5cd5eec4f4761d96",
    },
    ModelFile {
        remote: "onnx/vision_encoder_q4.onnx_data",
        local: "vision_encoder_q4.onnx_data",
        sha256: "0a9d6c927334f152a33dd90874f65d6ea5228999abe6a450d3f7813677fa704c",
    },
];

pub fn model_is_ready(directory: &Path) -> bool {
    MODEL_FILES
        .iter()
        .all(|file| directory.join(file.local).is_file())
}

pub fn download_model(
    directory: &Path,
    mut report: impl FnMut(&str, u64, Option<u64>),
) -> Result<()> {
    fs::create_dir_all(directory)?;
    let client = reqwest::blocking::Client::builder()
        .user_agent(concat!("LumenFind/", env!("CARGO_PKG_VERSION")))
        .build()?;

    for model_file in MODEL_FILES {
        let destination = directory.join(model_file.local);
        if destination.is_file() && sha256_file(&destination)? == model_file.sha256 {
            continue;
        }
        let url = format!(
            "https://huggingface.co/{REPOSITORY}/resolve/{REVISION}/{}",
            model_file.remote
        );
        let mut response = client.get(url).send()?.error_for_status()?;
        let total = response.content_length();
        let temporary = destination.with_extension("download");
        let mut output = File::create(&temporary)?;
        let mut hasher = Sha256::new();
        let mut buffer = vec![0_u8; 1024 * 1024];
        let mut received = 0;
        loop {
            let count = response.read(&mut buffer)?;
            if count == 0 {
                break;
            }
            output.write_all(&buffer[..count])?;
            hasher.update(&buffer[..count]);
            received += count as u64;
            report(model_file.local, received, total);
        }
        output.sync_all()?;
        let actual = format!("{:x}", hasher.finalize());
        if actual != model_file.sha256 {
            let _ = fs::remove_file(&temporary);
            bail!("checksum mismatch for {}", model_file.local);
        }
        fs::rename(temporary, destination)?;
    }
    Ok(())
}

fn sha256_file(path: &Path) -> Result<String> {
    let mut input = File::open(path)?;
    let mut hasher = Sha256::new();
    std::io::copy(&mut input, &mut hasher)?;
    Ok(format!("{:x}", hasher.finalize()))
}

pub struct EmbeddingModel {
    _environment: Environment,
    tokenizer: Tokenizer,
    text_model: Session,
    vision_model: Session,
}

impl EmbeddingModel {
    pub fn load(directory: &Path) -> Result<Self> {
        if !model_is_ready(directory) {
            bail!("EmbeddingGemma 2 has not been downloaded yet");
        }
        let tokenizer = Tokenizer::from_file(directory.join("tokenizer.json"))
            .map_err(|error| anyhow!(error.to_string()))?;
        static ENVIRONMENT: OnceLock<Result<Environment, String>> = OnceLock::new();
        let environment = ENVIRONMENT
            .get_or_init(|| {
                ort::init()
                    .with_name("LumenFind")
                    .with_telemetry(false)
                    .build()
                    .map_err(|error| error.to_string())
            })
            .as_ref()
            .map_err(|error| anyhow!(error.clone()))?
            .clone();
        let text_model = Session::builder(&environment)
            .map_err(|error| anyhow!(error.to_string()))?
            .with_intra_threads(2)
            .map_err(|error| anyhow!(error.to_string()))?
            .commit_from_file(directory.join("model_q4.onnx"))
            .map_err(|error| anyhow!(error.to_string()))?;
        let vision_model = Session::builder(&environment)
            .map_err(|error| anyhow!(error.to_string()))?
            .with_intra_threads(2)
            .map_err(|error| anyhow!(error.to_string()))?
            .commit_from_file(directory.join("vision_encoder_q4.onnx"))
            .map_err(|error| anyhow!(error.to_string()))?;
        Ok(Self {
            _environment: environment,
            tokenizer,
            text_model,
            vision_model,
        })
    }

    pub fn embed_query(&mut self, query: &str) -> Result<Vec<f32>> {
        let prompted = format!("task: search result | query: {}", query.trim());
        let encoding = self
            .tokenizer
            .encode(prompted, true)
            .map_err(|error| anyhow!(error.to_string()))?;
        let ids: Vec<i64> = encoding
            .get_ids()
            .iter()
            .map(|&value| value as i64)
            .collect();
        self.run_text(ids, Array2::zeros((0, 512)))
    }

    pub fn embed_image_path(&mut self, path: &Path) -> Result<Vec<f32>> {
        let image = load_image(path)?;
        self.embed_image(&image)
    }

    pub fn embed_image(&mut self, image: &DynamicImage) -> Result<Vec<f32>> {
        let prepared = prepare_image(image)?;
        let outputs = self.vision_model.run(ort::inputs![
            "pixel_values" => Tensor::from_array(prepared.patches)?,
            "pixel_position_ids" => Tensor::from_array(prepared.positions)?,
        ])?;
        let features = outputs["image_features"].try_extract_array::<f32>()?;
        let flattened: Vec<f32> = features.iter().copied().collect();
        if !flattened.len().is_multiple_of(512) {
            bail!("vision encoder returned an invalid feature shape");
        }
        let soft_tokens = flattened.len() / 512;
        if soft_tokens != prepared.soft_tokens {
            bail!(
                "vision encoder returned {soft_tokens} tokens, expected {}",
                prepared.soft_tokens
            );
        }
        let features = Array2::from_shape_vec((soft_tokens, 512), flattened)?;
        drop(outputs);
        let mut ids = Vec::with_capacity(soft_tokens + 3);
        ids.push(BOS_TOKEN);
        ids.push(BEGIN_IMAGE_TOKEN);
        ids.extend(std::iter::repeat_n(IMAGE_TOKEN, soft_tokens));
        ids.push(END_IMAGE_TOKEN);
        self.run_text(ids, features)
    }

    fn run_text(&mut self, ids: Vec<i64>, image_features: Array2<f32>) -> Result<Vec<f32>> {
        let length = ids.len();
        let input_ids = Array2::from_shape_vec((1, length), ids)?;
        let attention_mask = Array2::from_elem((1, length), 1_i64);
        let outputs = self.text_model.run(ort::inputs![
            "input_ids" => Tensor::from_array(input_ids)?,
            "attention_mask" => Tensor::from_array(attention_mask)?,
            "image_features" => Tensor::from_array(image_features)?,
            "video_features" => Tensor::from_array(Array2::<f32>::zeros((0, 512)))?,
            "audio_features" => Tensor::from_array(Array2::<f32>::zeros((0, 512)))?,
        ])?;
        let output = outputs["sentence_embedding"].try_extract_array::<f32>()?;
        let mut embedding: Vec<f32> = output.iter().copied().collect();
        if embedding.len() != EMBEDDING_DIM {
            bail!("model returned {} dimensions", embedding.len());
        }
        normalize(&mut embedding)?;
        Ok(embedding)
    }
}

struct PreparedImage {
    patches: Array3<f32>,
    positions: Array3<i64>,
    soft_tokens: usize,
}

fn prepare_image(image: &DynamicImage) -> Result<PreparedImage> {
    let (width, height) = image.dimensions();
    let (target_width, target_height) = aspect_preserving_size(width, height)?;
    let rgb = image
        .resize_exact(target_width, target_height, FilterType::CatmullRom)
        .to_rgb8();
    let patch_width = target_width as usize / PATCH_SIZE;
    let patch_height = target_height as usize / PATCH_SIZE;
    let patch_count = patch_width * patch_height;
    let soft_tokens = patch_count / (POOLING * POOLING);
    let mut patches = vec![0.0_f32; MAX_PATCHES * PATCH_SIZE * PATCH_SIZE * 3];
    let mut positions = vec![-1_i64; MAX_PATCHES * 2];

    for patch_y in 0..patch_height {
        for patch_x in 0..patch_width {
            let patch_index = patch_y * patch_width + patch_x;
            positions[patch_index * 2] = patch_x as i64;
            positions[patch_index * 2 + 1] = patch_y as i64;
            let base = patch_index * PATCH_SIZE * PATCH_SIZE * 3;
            for inner_y in 0..PATCH_SIZE {
                for inner_x in 0..PATCH_SIZE {
                    let pixel = rgb.get_pixel(
                        (patch_x * PATCH_SIZE + inner_x) as u32,
                        (patch_y * PATCH_SIZE + inner_y) as u32,
                    );
                    let pixel_base = base + (inner_y * PATCH_SIZE + inner_x) * 3;
                    patches[pixel_base] = pixel[0] as f32 / 255.0;
                    patches[pixel_base + 1] = pixel[1] as f32 / 255.0;
                    patches[pixel_base + 2] = pixel[2] as f32 / 255.0;
                }
            }
        }
    }

    Ok(PreparedImage {
        patches: Array3::from_shape_vec((1, MAX_PATCHES, PATCH_SIZE * PATCH_SIZE * 3), patches)?,
        positions: Array3::from_shape_vec((1, MAX_PATCHES, 2), positions)?,
        soft_tokens,
    })
}

fn aspect_preserving_size(width: u32, height: u32) -> Result<(u32, u32)> {
    if width == 0 || height == 0 {
        bail!("image has an empty dimension");
    }
    let target_pixels = (MAX_PATCHES * PATCH_SIZE * PATCH_SIZE) as f64;
    let factor = (target_pixels / (width as f64 * height as f64)).sqrt();
    let multiple = (PATCH_SIZE * POOLING) as u32;
    let mut target_width = ((width as f64 * factor).floor() as u32 / multiple) * multiple;
    let mut target_height = ((height as f64 * factor).floor() as u32 / multiple) * multiple;
    let max_side = (MAX_PATCHES / (POOLING * POOLING)) as u32 * multiple;
    if target_width == 0 {
        target_width = multiple;
        target_height = (((height / width).max(1) * multiple).min(max_side)).max(multiple);
    }
    if target_height == 0 {
        target_height = multiple;
        target_width = (((width / height).max(1) * multiple).min(max_side)).max(multiple);
    }
    Ok((target_width, target_height))
}

fn normalize(vector: &mut [f32]) -> Result<()> {
    let norm = vector.iter().map(|value| value * value).sum::<f32>().sqrt();
    if !norm.is_finite() || norm <= f32::EPSILON {
        bail!("model returned a zero or non-finite embedding");
    }
    for value in vector {
        *value /= norm;
    }
    Ok(())
}

pub fn required_download_bytes() -> u64 {
    32_170_510 + 490_742 + 174_028_800 + 159_400 + 108_957_696
}

pub fn model_files(directory: &Path) -> Vec<PathBuf> {
    MODEL_FILES
        .iter()
        .map(|file| directory.join(file.local))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resize_respects_patch_budget_and_pooling_grid() {
        for (width, height) in [(4000, 3000), (300, 4000), (4000, 300)] {
            let (w, h) = aspect_preserving_size(width, height).unwrap();
            assert_eq!(w % 48, 0);
            assert_eq!(h % 48, 0);
            assert!((w as usize / 16) * (h as usize / 16) <= MAX_PATCHES);
        }
    }

    #[test]
    fn normalization_produces_unit_vector() {
        let mut value = vec![3.0, 4.0];
        normalize(&mut value).unwrap();
        assert!((value.iter().map(|x| x * x).sum::<f32>() - 1.0).abs() < 1e-6);
    }
}
