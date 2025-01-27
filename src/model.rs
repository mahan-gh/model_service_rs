use std::fs::File;
use std::io::Read;

use image::{DynamicImage, GenericImageView, ImageBuffer, Rgb};
use serde::{Deserialize, Serialize};
use tensorflow::{
    Graph, ImportGraphDefOptions, Operation, Session, SessionOptions, SessionRunArgs, Tensor,
};

const IMAGE_DIMENSIONS: (u64, u64, u64) = (600, 600, 3);

#[derive(Serialize, Deserialize)]
pub struct Prediction {
    class_label: String,
    probability: f32,
}

pub struct Model {
    session: Session,
    labels: Vec<String>,
    input_layer: Operation,
    output_layer: Operation,
}

impl Model {
    pub fn new(model_path: &str, labels_path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let mut graph = Graph::new();
        let mut model_file = File::open(model_path)?;
        let mut model_bytes = Vec::new();
        model_file.read_to_end(&mut model_bytes)?;

        graph.import_graph_def(&model_bytes, &ImportGraphDefOptions::new())?;

        let session = Session::new(&SessionOptions::new(), &graph)?;

        let labels: Vec<String> = std::fs::read_to_string(labels_path)?
            .lines()
            .map(String::from)
            .collect();

        let input_layer_name = graph
            .operation_iter()
            .next()
            .ok_or("No operations found in the graph")?
            .name()
            .map_err(|_| "Failed to retrieve the operation name")?;

        println!("found {} as input layer", input_layer_name);

        let input_layer = graph
            .operation_by_name(&input_layer_name)?
            .ok_or_else(|| format!("Output operation '{}' not found in graph", input_layer_name))?;

        let output_layer_name = graph
            .operation_iter()
            .last()
            .ok_or("No operations found in the graph")?
            .name()
            .map_err(|_| "Failed to retrieve the operation name")?;

        println!("found {} as output layer", output_layer_name);

        let output_layer = graph
            .operation_by_name(&output_layer_name)?
            .ok_or_else(|| {
                format!(
                    "Output operation '{}' not found in graph",
                    output_layer_name
                )
            })?;

        Ok(Model {
            session,
            labels,
            input_layer,
            output_layer,
        })
    }

    fn resize_image_aspect_ratio(&self, image: &DynamicImage, target_size: u32) -> DynamicImage {
        let (original_width, original_height) = image.dimensions();
        let aspect_ratio = original_width as f32 / original_height as f32;

        // a bit different than my python implementation but seems like this is the correct way
        if aspect_ratio >= 1.0 {
            // Wider images
            let new_height = (target_size as f32 * aspect_ratio).floor() as u32;
            image.resize(
                target_size,
                new_height,
                image::imageops::FilterType::Lanczos3,
            )
        } else {
            // Taller images
            let new_width = (target_size as f32 / aspect_ratio).floor() as u32;
            image.resize(
                new_width,
                target_size,
                image::imageops::FilterType::Lanczos3,
            )
        }
    }

    fn center_crop_tall_image(&self, image: &DynamicImage) -> DynamicImage {
        let (original_width, original_height) = image.dimensions();

        // Calculate crop height (80% of original height)
        let crop_height = (0.8 * original_height as f32).round() as u32;
        let offset_height = (original_height - crop_height) / 2;

        image.crop_imm(0, offset_height, original_width, crop_height)
    }

    fn pad_to_square(
        &self,
        image: &DynamicImage,
        target_size: u32,
    ) -> ImageBuffer<Rgb<u8>, Vec<u8>> {
        let (width, height) = image.dimensions();
        let mut output = ImageBuffer::from_pixel(target_size, target_size, image::Rgb([0, 0, 0]));
        let rgb_image = image.to_rgb8();

        let x_offset = ((target_size - width) / 2).max(0);
        let y_offset = ((target_size - height) / 2).max(0);

        // Overlay the image directly onto the pre-filled buffer
        image::imageops::overlay(&mut output, &rgb_image, x_offset as i64, y_offset as i64);

        output
    }

    fn preprocess_image(
        &self,
        image_data: &[u8],
    ) -> Result<Tensor<f32>, Box<dyn std::error::Error>> {
        let mut img = image::load_from_memory(image_data)?;

        if img.color() == image::ColorType::L8 {
            img = DynamicImage::ImageRgb8(img.to_rgb8());
        }

        let (original_width, original_height) = img.dimensions();
        if original_height > 2 * original_width {
            img = self.center_crop_tall_image(&img);
        }

        img = self.resize_image_aspect_ratio(&img, IMAGE_DIMENSIONS.0 as u32);
        let padded = self.pad_to_square(&img, IMAGE_DIMENSIONS.0 as u32);

        // save_image_to_folder(&padded);

        let mut tensor = Tensor::new(&[
            1,
            IMAGE_DIMENSIONS.0,
            IMAGE_DIMENSIONS.1,
            IMAGE_DIMENSIONS.2,
        ]);

        tensor
            .as_mut()
            .chunks_mut(IMAGE_DIMENSIONS.2.try_into().unwrap())
            .zip(padded.pixels())
            .for_each(|(tensor_chunk, pixel)| {
                tensor_chunk[0] = pixel[0] as f32;
                tensor_chunk[1] = pixel[1] as f32;
                tensor_chunk[2] = pixel[2] as f32;
            });

        Ok(tensor)
    }

    pub fn predict(
        &self,
        image_data: &[u8],
    ) -> Result<Vec<Prediction>, Box<dyn std::error::Error>> {
        let input_tensor = self.preprocess_image(image_data)?;

        let mut args = SessionRunArgs::new();

        args.add_feed(&self.input_layer, 0, &input_tensor);
        let output_token = args.request_fetch(&self.output_layer, 0);
        self.session.run(&mut args)?;

        let output_tensor: tensorflow::Tensor<f32> = args.fetch(output_token)?;
        let predictions: Vec<f32> = output_tensor.to_vec();

        let mut prediction_result: Vec<Prediction> = predictions
            .iter()
            .enumerate()
            .filter(|(_, &prob)| (prob * 10000.0).round() > 0.0)
            .map(|(i, &prob)| {
                // let percentage = (prob * 100.0 * 100.0).round() / 100.0;
                Prediction {
                    class_label: self
                        .labels
                        .get(i)
                        .cloned()
                        .unwrap_or_else(|| "Unknown".to_string()),
                    probability: (prob * 100.0),
                }
            })
            .collect();

        prediction_result.sort_by(|a, b| {
            b.probability
                .partial_cmp(&a.probability)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(prediction_result)
    }
}

// use std::path::Path;
// use std::time::{SystemTime, UNIX_EPOCH};

// const PATH: &str = r"C:\Users\Mahan\Desktop\image_rs";

// fn save_image_to_folder(image: &ImageBuffer<Rgb<u8>, Vec<u8>>) -> Result<(), String> {
//     // Ensure the folder exists
//     let folder_path = Path::new(PATH);
//     // if let Err(e) = fs::create_dir_all(folder_path) {
//     //     return Err(format!("Failed to create folder: {}", e));
//     // }

//     // Generate a unique filename using a timestamp
//     let timestamp = SystemTime::now()
//         .duration_since(UNIX_EPOCH)
//         .map_err(|e| format!("Failed to get system time: {}", e))?
//         .as_secs();
//     let filename = format!("image_{}.jpg", timestamp);

//     // Construct the full path for the image
//     let file_path = folder_path.join(filename);

//     // Write the image to the specified path
//     if let Err(e) = image.save(&file_path) {
//         return Err(format!("Failed to save image: {}", e));
//     }

//     Ok(())
// }
