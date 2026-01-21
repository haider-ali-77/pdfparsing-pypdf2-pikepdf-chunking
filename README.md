# PDF Parsing with PDFConverter

`PDFConverter` is a Python class designed to parse PDFs, extract text using **GROBID**, and convert PDF pages into images. This Markdown includes everything: installation, class code, usage examples, and requirements.

---

## Installation

1. Make sure you have Python 3.7+ installed.
2. Install required dependencies:

```bash
pip install requirements.txt
```

### Sample use:
```
from typing import Optional
from grobid_client.grobid_client import GrobidClient
from your_module import PDFConverter  # Replace with your module name

# Initialize PDFConverter
pdf_converter = PDFConverter(
    grobid_client=GrobidClient(config_path="config.json"),  # Optional
    image_density=300,      # DPI for images
    image_depth=8,          # Bit depth of images
    image_format="png"      # Image format
)
```
