use std::{env, fs, path::Path};

use reqwest::header::{HeaderMap, HeaderName, HeaderValue};

async fn download_file(url: &str, path: &str) {
    println!("Downloading {} from {}", path, url);

    let mut header_map = HeaderMap::new();

    if let Ok(token) = env::var("GITHUB_TOKEN") {
        let auth_value = HeaderValue::from_str(&format!("Bearer {}", token))
            .expect("Invalid GITHUB_TOKEN format");
        header_map.insert(HeaderName::from_static("Authorization"), auth_value);
    }
    header_map.insert(
        HeaderName::from_static("accept"),
        HeaderValue::from_static("application/octet-stream"),
    );

    let client = reqwest::Client::new();
    let response = client
        .get(url)
        .headers(header_map)
        .send()
        .await
        .expect("Failed to send request");

    if !response.status().is_success() {
        panic!("Failed to download {}: {}", url, response.status());
    }

    // Write file
    let bytes = response.bytes().await.expect("Failed to read bytes");
    fs::write(path, bytes).expect("Failed to write file");
}

pub async fn ensure_files_exist(model_path: &str, class_list_path: &str) {
    println!("Checking model...");
    if !Path::new(model_path).exists() {
        let model_url = env::var("MODEL_URL").expect("MODEL_URL environment variable not set");
        download_file(&model_url, model_path).await;
    }

    if !Path::new(class_list_path).exists() {
        let class_url =
            env::var("CLASS_LIST_URL").expect("CLASS_LIST_URL environment variable not set");
        download_file(&class_url, class_list_path).await;
    }
}

pub fn get_env() -> (usize, u16) {
    let body_limit_bytes = {
        let mb = env::var("BODY_LIMIT_MB")
            .unwrap_or_else(|_| "5".into())
            .parse::<usize>()
            .expect("BODY_LIMIT_MB must be a valid integer");
        mb * 1024 * 1024
    };

    let port = env::var("PORT")
        .unwrap_or_else(|_| "5020".into())
        .parse::<u16>()
        .expect("PORT must be a valid number between 0 and 65535");

    (body_limit_bytes, port)
}
