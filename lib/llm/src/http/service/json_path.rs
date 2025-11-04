use axum::{
    body::{to_bytes, Body},
    extract::FromRequest,
    http::{Request, StatusCode},
    response::IntoResponse,
};
use serde::de::DeserializeOwned;
use serde_json::Deserializer;

#[derive(Debug)]
pub struct JsonPath<T>(pub T);

impl<S, T> FromRequest<S, Body> for JsonPath<T>
where
    T: DeserializeOwned,
    S: Send + Sync,
{
    type Rejection = axum::response::Response;

    async fn from_request(req: Request<Body>, _state: &S) -> Result<Self, Self::Rejection> {
        let bytes = to_bytes(req.into_body(), usize::MAX)
            .await
            .map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()).into_response())?;

        match serde_path_to_error::deserialize(&mut Deserializer::from_slice(&bytes)) {
            Ok(v) => Ok(JsonPath(v)),
            Err(e) => {
                let field = e.path().to_string().trim_start_matches('.').to_string();
                let msg = format!(
                    "Invalid argument for parameter '{}': {}",
                    if field.is_empty() { "request body" } else { &field },
                    e.inner()
                );

                Err((
                    StatusCode::BAD_REQUEST,
                    axum::Json(serde_json::json!({
                        "message": msg,
                        "type": "Bad Request",
                        "code": 400
                    })),
                )
                    .into_response())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::Deserialize;

    #[derive(Deserialize, Debug)]
    struct TestRequest {
        model: String,
        temperature: Option<f32>,
        echo: Option<bool>,
    }

    #[tokio::test]
    async fn test_valid_request() {
        let body = r#"{"model": "gpt-4", "temperature": 0.7, "echo": true}"#;
        let req = Request::builder().body(Body::from(body)).unwrap();

        let result = JsonPath::<TestRequest>::from_request(req, &()).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_echo_as_number_shows_field_path() {
        let body = r#"{"model": "gpt-4", "echo": 1}"#;
        let req = Request::builder().body(Body::from(body)).unwrap();

        let result = JsonPath::<TestRequest>::from_request(req, &()).await;
        assert!(result.is_err());

        let response = result.unwrap_err();
        let body_bytes = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let body_str = String::from_utf8(body_bytes.to_vec()).unwrap();

        assert!(body_str.contains("echo"));
        assert!(body_str.contains("Invalid argument for parameter"));
    }

    #[tokio::test]
    async fn test_temperature_as_string_shows_field_path() {
        let body = r#"{"model": "gpt-4", "temperature": "hot"}"#;
        let req = Request::builder().body(Body::from(body)).unwrap();

        let result = JsonPath::<TestRequest>::from_request(req, &()).await;
        assert!(result.is_err());

        let response = result.unwrap_err();
        let body_bytes = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let body_str = String::from_utf8(body_bytes.to_vec()).unwrap();

        assert!(body_str.contains("temperature"));
    }
}
