# AI Paper Keyword Extractor

A lightweight Python tool that extracts the most important keywords and topics from research papers in PDF format. It works by reading the text from papers, cleaning it using natural language processing techniques, and applying keyword extraction algorithms such as TF-IDF and RAKE.

## Features

- **PDF Text Extraction**: Extracts text content from PDF research papers
- **Advanced Text Processing**: Cleans and preprocesses text for optimal keyword extraction
- **Multiple Algorithms**: Implements both TF-IDF and RAKE algorithms for comprehensive keyword extraction
- **Academic-Focused**: Specifically designed for research papers with academic stop words filtering
- **Flexible Output**: Supports text, JSON, and CSV output formats
- **Batch Processing**: Process multiple PDFs at once
- **Offline Operation**: Works entirely offline using free, open-source libraries

## Benefits

- **Time-Saving**: Quickly identify the main topics of research papers without reading them fully
- **Research Organization**: Tag and categorize papers for building research databases
- **Topic Discovery**: Discover key concepts and trending topics in academic literature
- **Content Analysis**: Analyze the focus and scope of research papers

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone or download the project**
```bash
cd "AI Paper Keyword Extractor"
```

2. **Create and activate virtual environment**
```bash
# Create virtual environment
python3 -m venv keyword_extractor

# Activate virtual environment
# On macOS/Linux:
source keyword_extractor/bin/activate

# On Windows:
keyword_extractor\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Download NLTK data (automatic on first run)**
The application will automatically download required NLTK data packages on first use.

## Usage

### Command Line Interface

#### Basic Usage

Extract keywords from a single PDF:
```bash
python main.py paper.pdf
```

Extract more keywords with different output:
```bash
python main.py paper.pdf --keywords 20 --save --format json
```

Process multiple PDFs in a directory:
```bash
python main.py --directory ./research_papers --keywords 15 --save
```

#### Command Line Options

```
python main.py [PDF_FILE] [OPTIONS]

positional arguments:
  pdf_file              Path to PDF file to process

options:
  -h, --help            Show help message
  --directory DIRECTORY, -d DIRECTORY
                        Directory containing PDF files to process
  --keywords KEYWORDS, -k KEYWORDS
                        Number of keywords to extract (default: 15)
  --save, -s            Save results to file
  --format {text,json,csv}, -f {text,json,csv}
                        Output format (default: text)
  --version             Show version number
```

### Examples

#### Example 1: Basic keyword extraction
```bash
python main.py research_paper.pdf
```

Output:
```
Processing PDF: research_paper.pdf
Step 1: Extracting text from PDF...
Extracted 45234 characters
Step 2: Preprocessing text...
Processed to 5432 words (2156 unique)
Step 3: Extracting keywords...

============================================================
KEYWORD EXTRACTION RESULTS
============================================================

🔑 RECOMMENDED KEYWORDS:
----------------------------------------
 1. machine learning          [combined]
 2. neural networks           [combined]
 3. deep learning             [combined]
 4. artificial intelligence   [combined]
 5. natural language          [combined]
...
```

#### Example 2: Save results in JSON format
```bash
python main.py paper.pdf --keywords 20 --save --format json
```

#### Example 3: Process multiple papers
```bash
python main.py --directory ./papers --keywords 10 --save
```

### Using as a Python Module

```python
from pdf_reader import PDFReader
from text_processor import TextProcessor
from keyword_extractor import KeywordExtractor

# Initialize components
pdf_reader = PDFReader()
text_processor = TextProcessor()
keyword_extractor = KeywordExtractor()

# Extract text from PDF
text = pdf_reader.extract_text("paper.pdf")

# Preprocess text
processed_data = text_processor.preprocess_for_keywords(text)

# Extract keywords
keywords = keyword_extractor.get_keyword_summary(text, processed_data)

print(keywords['recommended_keywords'])
```

## Output Formats

### Text Format (Default)
Human-readable text output with organized sections for different keyword extraction methods.

### JSON Format
Structured JSON output containing:
- PDF metadata
- Text statistics
- Complete keyword results from all algorithms
- Timestamps and processing information

### CSV Format
Tabular format suitable for spreadsheet applications with columns for rank, keyword, score, and extraction method.

## Algorithms

### TF-IDF (Term Frequency-Inverse Document Frequency)
- Identifies important terms based on their frequency in the document relative to their frequency across a corpus
- Good for finding domain-specific technical terms
- Weights: Higher scores for terms that appear frequently in the document but rarely elsewhere

### RAKE (Rapid Automatic Keyword Extraction)
- Extracts multi-word phrases and key terms
- Excellent for capturing compound concepts and technical phrases
- Uses word co-occurrence and phrase patterns

### Combined Approach
- Merges results from both TF-IDF and RAKE algorithms
- Provides balanced keyword extraction leveraging strengths of both methods
- Includes configurable weighting for different algorithm contributions

## Architecture

```
├── main.py                 # Main CLI application
├── pdf_reader.py          # PDF text extraction
├── text_processor.py      # Text cleaning and preprocessing
├── keyword_extractor.py   # Keyword extraction algorithms
├── requirements.txt       # Python dependencies
├── test_sample.py        # Test script with sample data
└── README.md            # Documentation
```

## Technical Details

### Text Preprocessing
- Removes citations, references, and academic formatting
- Filters out common academic stop words
- Handles mathematical expressions and equations
- Lemmatization and tokenization
- Noun phrase extraction

### Academic Paper Optimization
- Custom stop word lists for academic content
- Filters section headers and common paper elements
- Optimized for scientific and technical terminology
- Handles multi-word technical terms

## Testing

Run the test script to verify installation:
```bash
python test_sample.py
```

This will test the keyword extraction functionality with sample academic text and display the results.

## Troubleshooting

### Common Issues

1. **PDF Text Extraction Issues**
   - Ensure PDF contains extractable text (not just images)
   - Some PDFs may have security restrictions

2. **NLTK Data Missing**
   - The application automatically downloads required NLTK data
   - If issues persist, manually run: `python -c "import nltk; nltk.download('all')"`

3. **Memory Issues with Large PDFs**
   - For very large PDFs, consider processing in smaller chunks
   - Increase system memory or use a more powerful machine

### Performance Tips

- For better performance with large documents, increase the `max_features` parameter in TF-IDF
- Adjust `min_length` and `max_length` in RAKE for different phrase lengths
- Use batch processing for multiple documents to amortize initialization costs

## Dependencies

- **PyPDF2**: PDF text extraction
- **NLTK**: Natural language processing toolkit
- **scikit-learn**: TF-IDF vectorization and machine learning utilities
- **rake-nltk**: RAKE algorithm implementation
- **numpy**: Numerical operations

## System Requirements

- Python 3.8+
- 2GB RAM minimum (4GB recommended for large documents)
- 500MB disk space for dependencies and NLTK data

## License

Copyright (c) 2024 Sreeram Lagisetty. All rights reserved.

This project is proprietary software. Unauthorized copying, distribution, or use of this software is strictly prohibited.

## Contributing

Contributions are welcome! Areas for improvement:
- Support for additional file formats (DOC, DOCX, TXT)
- Advanced visualization of keyword relationships
- Machine learning-based keyword extraction
- Multi-language support
- Web interface

## Future Enhancements

- [ ] Support for other document formats
- [ ] Keyword visualization and word clouds
- [ ] Topic modeling integration
- [ ] Web-based interface
- [ ] API endpoints for integration
- [ ] Advanced filtering and customization options

---

**Happy researching!** This tool will help you quickly understand and organize your research paper collections.
