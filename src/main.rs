use std::sync::Arc;

use axum::{
    extract::{DefaultBodyLimit, Multipart, State},
    routing::{get, post},
    Json, Router,
};
use serde_json::json;
use tokio::sync::Mutex;

mod model;
use crate::model::Model;

const PORT: u16 = 8080;

struct AppState {
    model: Model,
}

#[tokio::main]
async fn main() {
    let model = Model::new("./frozen_graph.pb", "./class_list.txt").expect("Failed to load model");
    let shared_state = Arc::new(Mutex::new(AppState { model }));

    let app = Router::new()
        .route("/predict", post(predict_handler))
        .layer(DefaultBodyLimit::max(10 * 1024 * 1024)) // 10 MB limit
        .with_state(shared_state)
        .route("/health", get(health_check));

    println!("Listening on http://0.0.0.0:{}", PORT);
    axum::Server::bind(&format!("0.0.0.0:{}", PORT).parse().unwrap())
        .serve(app.into_make_service())
        .await
        .unwrap();
}

async fn predict_handler(
    State(state): State<Arc<Mutex<AppState>>>,
    mut multipart: Multipart,
) -> Json<serde_json::Value> {
    let mut image_data = Vec::new();

    // Process multipart form to find the file
    while let Some(field) = multipart.next_field().await.unwrap() {
        if field.name() == Some("file") {
            image_data = field.bytes().await.unwrap().to_vec();
            break;
        }
    }

    if image_data.is_empty() {
        return Json(json!({ "error": "No file uploaded" }));
    }

    let state = state.lock().await;
    let model = &state.model;

    match model.predict(&image_data) {
        Ok(result) => Json(json!({"prediction": result})),
        Err(err) => Json(json!({ "error": err.to_string() })),
    }
}

async fn health_check() -> Json<serde_json::Value> {
    Json(serde_json::json!({ "status": "OK" }))
}
