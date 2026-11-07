//! Pasting images: the clipboard image is saved as a PNG file and referenced in the prompt as
//! `[image: <path>]`, which the core attaches for the model.

use std::path::{Path, PathBuf};

const IMAGE_EXTENSIONS: [&str; 5] = ["png", "jpg", "jpeg", "gif", "webp"];

/// The `[image: …]` marker for `path`.
pub fn marker(path: &Path) -> String {
    format!("[image: {}]", path.display())
}

/// Saves the clipboard image under `dir` and returns the file.
pub fn save_clipboard_image(dir: &Path) -> Result<PathBuf, String> {
    let mut clipboard = arboard::Clipboard::new().map_err(|e| e.to_string())?;
    let image = clipboard.get_image().map_err(|e| match e {
        arboard::Error::ContentNotAvailable => "the clipboard holds no image".to_owned(),
        e => e.to_string(),
    })?;
    std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_millis());
    let path = dir.join(format!("paste-{stamp}.png"));
    let file = std::fs::File::create(&path).map_err(|e| e.to_string())?;
    let width = u32::try_from(image.width).map_err(|e| e.to_string())?;
    let height = u32::try_from(image.height).map_err(|e| e.to_string())?;
    let mut encoder = png::Encoder::new(std::io::BufWriter::new(file), width, height);
    encoder.set_color(png::ColorType::Rgba);
    encoder.set_depth(png::BitDepth::Eight);
    encoder
        .write_header()
        .and_then(|mut w| w.write_image_data(&image.bytes))
        .map_err(|e| e.to_string())?;
    Ok(path)
}

/// The clipboard's text, if any.
pub fn text() -> Option<String> {
    arboard::Clipboard::new()
        .ok()?
        .get_text()
        .ok()
        .filter(|t| !t.is_empty())
}

/// A pasted text that is just the path of an existing image file (quotes allowed).
pub fn pasted_image_path(text: &str) -> Option<PathBuf> {
    let t = text.trim().trim_matches(['"', '\'']);
    if t.is_empty() || t.contains('\n') {
        return None;
    }
    let path = PathBuf::from(t);
    let ext = path.extension()?.to_string_lossy().to_ascii_lowercase();
    (IMAGE_EXTENSIONS.contains(&ext.as_str()) && path.is_file()).then_some(path)
}
