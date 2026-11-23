use std::env;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("cargo:rerun-if-changed=src/opencv_buffer_capture.cpp");
    println!("cargo:rerun-if-env-changed=OPENCV_INCLUDE_PATHS");
    if env::var_os("CARGO_FEATURE_OPENCV_VIDEO").is_none() {
        return Ok(());
    }

    let mut build = cc::Build::new();
    build
        .cpp(true)
        .file("src/opencv_buffer_capture.cpp")
        .flag_if_supported("-std=c++17");

    if let Some(paths) = env::var_os("OPENCV_INCLUDE_PATHS") {
        for path in env::split_paths(&paths) {
            build.include(path);
        }
    } else {
        let opencv = pkg_config::Config::new()
            .cargo_metadata(false)
            .probe("opencv4")?;
        for path in opencv.include_paths {
            build.include(path);
        }
    }

    build.compile("smg_opencv_buffer_capture");
    Ok(())
}
