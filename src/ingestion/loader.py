import os
import cv2
import numpy as np
from typing import List, Optional
from pdf2image import convert_from_path
from src.utils.helpers import load_config

config = load_config()
storage_cfg = config.get("storage", {})
prep_cfg = config.get("preprocessor", {})

DEFAULT_PAGES_DIR = os.path.abspath(storage_cfg.get("pages_dir", "./storage/pages"))
DEFAULT_ENHANCED_DIR = os.path.abspath(storage_cfg.get("enhanced_pages_dir", "./storage/enhanced_pages"))
DEFAULT_DPI = prep_cfg.get("pdf_dpi", 300)
CLAHE_CLIP = prep_cfg.get("clahe_clip_limit", 2.0)
CLAHE_GRID = tuple(prep_cfg.get("clahe_tile_grid_size", [8, 8]))

def render_pdf(pdf_path: str, dpi: int = DEFAULT_DPI, output_dir: Optional[str] = None) -> List[str]:
    """Renders multi-page PDFs to image files (JPEG)."""
    target_dir = output_dir or DEFAULT_PAGES_DIR
    os.makedirs(target_dir, exist_ok=True)
    
    # Locate poppler binary path
    poppler_path = os.path.abspath("./tools/poppler-26.09.0/Library/bin")
    if not os.path.exists(poppler_path):
        poppler_path = None
        
    pages = convert_from_path(pdf_path, dpi=dpi, poppler_path=poppler_path)
    output_paths = []
    
    base_name = os.path.splitext(os.path.basename(pdf_path))[0]
    for i, page in enumerate(pages):
        page_path = os.path.join(target_dir, f"{base_name}_page_{i+1}.jpg")
        page.save(page_path, 'JPEG')
        output_paths.append(page_path)
        
    return output_paths

def enhance_scan(image_path: str, output_dir: Optional[str] = None) -> str:
    """
    Crops non-document borders using contour bounding boxes and applies
    CLAHE contrast filtering to make faint text and handwriting clear.
    """
    target_dir = output_dir or DEFAULT_ENHANCED_DIR
    os.makedirs(target_dir, exist_ok=True)
    
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    # Crop non-document borders
    gray_for_crop = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray_for_crop, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        if w * h > 0.1 * (img.shape[0] * img.shape[1]):
            img = img[y:y+h, x:x+w]

    # Convert to grayscale for CLAHE
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_GRID)
    enhanced_gray = clahe.apply(gray)
    enhanced_rgb = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2RGB)
    
    base_name = os.path.basename(image_path)
    output_path = os.path.join(target_dir, f"enhanced_{base_name}")
    cv2.imwrite(output_path, enhanced_rgb)
    return output_path
