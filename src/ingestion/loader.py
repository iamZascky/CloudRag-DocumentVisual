import os
import sys

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENCV_CPU_MAX_THREADS"] = "1"

try:
    import cv2
    cv2.setNumThreads(1)
except Exception:
    cv2 = None

import numpy as np
from typing import List, Optional
from src.utils.helpers import load_config

config = load_config()
storage_cfg = config.get("storage", {})
prep_cfg = config.get("preprocessor", {})

DEFAULT_PAGES_DIR = os.path.abspath(storage_cfg.get("pages_dir", "./storage/pages"))
DEFAULT_ENHANCED_DIR = os.path.abspath(storage_cfg.get("enhanced_pages_dir", "./storage/enhanced_pages"))
DEFAULT_DPI = prep_cfg.get("pdf_dpi", 150)
CLAHE_CLIP = prep_cfg.get("clahe_clip_limit", 2.0)
CLAHE_GRID = tuple(prep_cfg.get("clahe_tile_grid_size", [8, 8]))

def render_pdf(pdf_path: str, dpi: int = DEFAULT_DPI, output_dir: Optional[str] = None) -> List[str]:
    """
    Renders multi-page PDFs to image files (JPEG).
    Supports Poppler (pdf2image) and falls back to pypdfium2 / pymupdf if poppler is absent.
    """
    target_dir = output_dir or DEFAULT_PAGES_DIR
    os.makedirs(target_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(pdf_path))[0]
    output_paths = []

    # Method 0 (preferred): pypdfium2 streaming - renders & frees ONE page at a time.
    # Keeps RAM flat (~30 MB) even for 100+ page PDFs on cPanel (1 GB LVE limit).
    try:
        import pypdfium2 as pdfium
        pdf = pdfium.PdfDocument(pdf_path)
        scale = dpi / 72.0
        for i in range(len(pdf)):
            page = pdf[i]
            bitmap = page.render(scale=scale)
            pil_image = bitmap.to_pil()
            page_path = os.path.join(target_dir, f"{base_name}_page_{i+1}.jpg")
            pil_image.save(page_path, 'JPEG', quality=90)
            output_paths.append(page_path)
            pil_image.close()
            bitmap.close()
            page.close()
        pdf.close()
        return output_paths
    except ImportError:
        pass
    except Exception as stream_err:
        print(f"[PDF Loader] pypdfium2 streaming failed ({stream_err}). Trying other renderers...")
        output_paths = []

    # Method 1: Try pdf2image with Poppler
    try:
        from pdf2image import convert_from_path
        poppler_path = os.path.abspath("./tools/poppler-26.09.0/Library/bin")
        if not os.path.exists(poppler_path):
            poppler_path = None
        pages = convert_from_path(pdf_path, dpi=dpi, poppler_path=poppler_path)
        for i, page in enumerate(pages):
            page_path = os.path.join(target_dir, f"{base_name}_page_{i+1}.jpg")
            page.save(page_path, 'JPEG')
            output_paths.append(page_path)
        return output_paths
    except Exception as poppler_err:
        print(f"[PDF Loader] Poppler conversion failed/absent ({poppler_err}). Trying fallback renderer...")

    # Method 2: Fallback to pypdfium2 (Pure python / no external poppler required)
    try:
        import pypdfium2 as pdfium
        pdf = pdfium.PdfDocument(pdf_path)
        scale = dpi / 72.0
        for i in range(len(pdf)):
            page = pdf[i]
            pil_image = page.render(scale=scale).to_pil()
            page_path = os.path.join(target_dir, f"{base_name}_page_{i+1}.jpg")
            pil_image.save(page_path, 'JPEG')
            output_paths.append(page_path)
        return output_paths
    except ImportError:
        pass
    except Exception as pdfium_err:
        print(f"[PDF Loader] pypdfium2 rendering error: {pdfium_err}")

    # Method 3: Fallback to PyMuPDF (fitz)
    try:
        import fitz
        doc = fitz.open(pdf_path)
        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        for i, page in enumerate(doc):
            pix = page.get_pixmap(matrix=mat)
            page_path = os.path.join(target_dir, f"{base_name}_page_{i+1}.jpg")
            pix.save(page_path)
            output_paths.append(page_path)
        return output_paths
    except ImportError:
        pass
    except Exception as fitz_err:
        print(f"[PDF Loader] PyMuPDF rendering error: {fitz_err}")

    raise RuntimeError(
        "No PDF renderer found or all renderers failed. Install pypdfium2 or PyMuPDF."
    )

def enhance_scan(image_path: str, output_dir: Optional[str] = None) -> str:
    """
    Lightweight image preparation for Vision AI.
    Saves an enhanced reference rapidly without heavy CPU-bound OpenCV filters.
    """
    target_dir = output_dir or DEFAULT_ENHANCED_DIR
    os.makedirs(target_dir, exist_ok=True)
    
    base_name = os.path.basename(image_path)
    output_path = os.path.join(target_dir, f"enhanced_{base_name}")
    
    # Fast path: digital PDF renders are already clear. Direct copy avoids 2-3min CPU throttling per page!
    import shutil
    try:
        shutil.copyfile(image_path, output_path)
        return output_path
    except Exception:
        pass

    # Fallback to OpenCV only if copy fails
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    cv2.imwrite(output_path, img)
    return output_path

def analyze_pdf_page_content(pdf_path: str, page_idx: int) -> dict:
    """
    Hybrid Completeness Detector:
    Analyzes whether a PDF page contains purely digital text or embedded raster images (receipts/stamps/scans).
    Returns:
    {
        "has_digital_text": bool,
        "text_length": int,
        "extracted_text": str,
        "has_embedded_images": bool,
        "image_count": int,
        "is_pure_digital": bool  # True ONLY if sufficient text and ZERO embedded photos/receipts
    }
    """
    result = {
        "has_digital_text": False,
        "text_length": 0,
        "extracted_text": "",
        "has_embedded_images": False,
        "image_count": 0,
        "is_pure_digital": False
    }

    # Method 1: Try pypdfium2
    try:
        import pypdfium2 as pdfium
        pdf = pdfium.PdfDocument(pdf_path)
        if 0 <= page_idx < len(pdf):
            page = pdf[page_idx]
            textpage = page.get_textpage()
            text = textpage.get_text_range().strip()
            
            # Count image objects on page
            img_count = 0
            try:
                for obj in page.get_objects():
                    if obj.type == pdfium.raw.FPDF_PAGEOBJ_IMAGE:
                        img_count += 1
            except Exception:
                pass

            result["text_length"] = len(text)
            result["extracted_text"] = text
            result["has_digital_text"] = len(text) > 80
            result["image_count"] = img_count
            result["has_embedded_images"] = img_count > 0
            # Pure digital only if substantial text and NO embedded raster photos/receipts
            result["is_pure_digital"] = result["has_digital_text"] and not result["has_embedded_images"]
            return result
    except Exception:
        pass

    # Method 2: Fallback to PyMuPDF (fitz)
    try:
        import fitz
        doc = fitz.open(pdf_path)
        if 0 <= page_idx < len(doc):
            page = doc[page_idx]
            text = page.get_text().strip()
            images = page.get_images()
            img_count = len(images)

            result["text_length"] = len(text)
            result["extracted_text"] = text
            result["has_digital_text"] = len(text) > 80
            result["image_count"] = img_count
            result["has_embedded_images"] = img_count > 0
            result["is_pure_digital"] = result["has_digital_text"] and not result["has_embedded_images"]
            return result
    except Exception:
        pass

    return result
