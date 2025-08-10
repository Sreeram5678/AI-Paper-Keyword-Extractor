# AI Paper Keyword Extractor - Usage Guide

## 🚀 Quick Start

### Basic Setup
```bash
# Navigate to project directory
cd "AI Paper Keyword Extractor"

# Activate virtual environment
source keyword_extractor/bin/activate

# Basic usage
python main.py your_paper.pdf
```

## 📋 Command Reference

### 1. Basic Extraction Commands

#### Extract keywords from a single PDF (default: 25 keywords)
```bash
python main.py paper.pdf
```

#### Extract keywords with custom count
```bash
python main.py paper.pdf --keywords 15
python main.py paper.pdf --keywords 30
python main.py paper.pdf --keywords 50
python main.py paper.pdf --keywords 100
```

#### Short form using -k flag
```bash
python main.py paper.pdf -k 40
```

### 2. Output and Saving Commands

#### Save results to file (text format)
```bash
python main.py paper.pdf --save
python main.py paper.pdf -s
```

#### Save in different formats
```bash
# Save as JSON (recommended for detailed analysis)
python main.py paper.pdf --save --format json
python main.py paper.pdf -s -f json

# Save as CSV (good for spreadsheets)
python main.py paper.pdf --save --format csv
python main.py paper.pdf -s -f csv

# Save as text (human-readable)
python main.py paper.pdf --save --format text
python main.py paper.pdf -s -f text
```

#### Combine keywords count with saving
```bash
python main.py paper.pdf --keywords 50 --save --format json
python main.py paper.pdf -k 50 -s -f json
```

### 3. Batch Processing Commands

#### Process all PDFs in a directory
```bash
python main.py --directory ./papers
python main.py -d ./papers
```

#### Batch process with custom settings
```bash
# Process directory with more keywords and save results
python main.py --directory ./papers --keywords 30 --save
python main.py -d ./papers -k 30 -s

# Batch process and save as JSON
python main.py --directory ./papers --keywords 25 --save --format json
python main.py -d ./papers -k 25 -s -f json
```

### 4. Advanced Usage Examples

#### High-detail analysis with comprehensive output
```bash
python main.py paper.pdf --keywords 100 --save --format json
```

#### Quick analysis with moderate detail
```bash
python main.py paper.pdf --keywords 20
```

#### Research workflow example
```bash
# Process all papers in research folder with detailed extraction
python main.py --directory ./research_papers --keywords 40 --save --format json
```

## 🎯 Specific Use Cases

### For Academic Researchers
```bash
# Analyze single paper with comprehensive keywords
python main.py "research_paper.pdf" --keywords 50 --save --format json

# Analyze entire literature collection
python main.py --directory "./literature_review" --keywords 30 --save
```

### For Quick Paper Screening
```bash
# Quick overview of paper topics
python main.py paper.pdf --keywords 15

# Screen multiple papers quickly
python main.py --directory ./new_papers --keywords 20
```

### For Building Research Databases
```bash
# Extract detailed keywords for database entry
python main.py paper.pdf --keywords 75 --save --format csv

# Batch process for research database
python main.py --directory ./database_papers --keywords 50 --save --format csv
```

## 📊 Understanding Output Sections

### 🔑 RECOMMENDED KEYWORDS
- **Best overall keywords** combining TF-IDF and RAKE algorithms
- **Count**: Controlled by `--keywords` parameter
- **Source indicated**: Shows which algorithm found each keyword

### 📊 TF-IDF KEYWORDS  
- **Technical terms** and domain-specific vocabulary
- **Shows 20 keywords** with relevance scores
- **Good for**: Finding specialized terminology

### 🔍 RAKE KEYWORDS
- **Multi-word phrases** and key concepts
- **Shows 20 keywords** with importance scores  
- **Good for**: Understanding main topics and concepts

### 📝 KEY PHRASES
- **Frequently occurring phrases** in the document
- **Shows 10 phrases** with frequency counts
- **Good for**: Identifying repeated themes

## 🔧 Customization Options

### Modify Default Keyword Count
The default is now set to 25 keywords. To permanently change it, edit `main.py` line 288:
```python
parser.add_argument('--keywords', '-k', type=int, default=25,  # Change this number
                   help='Number of keywords to extract (default: 25)')
```

### Increase Section Display Counts
To show more keywords in each section, modify these lines in `main.py`:
- Line 165: `summary['tfidf_keywords'][:20]` (change 20 to desired count)
- Line 170: `summary['rake_keywords'][:20]` (change 20 to desired count) 
- Line 176: `summary['noun_phrase_keywords'][:10]` (change 10 to desired count)

## 📁 File Output Examples

### Text Output Format
```
AI PAPER KEYWORD EXTRACTION RESULTS
==================================================

PDF File: paper.pdf
Extraction Time: 2024-01-09T14:30:45

RECOMMENDED KEYWORDS:
------------------------------
 1. machine learning [combined]
 2. neural networks [combined]
 ...
```

### JSON Output Format
```json
{
  "pdf_info": {
    "title": "Paper Title",
    "author": "Author Name", 
    "num_pages": 12
  },
  "keyword_summary": {
    "recommended_keywords": [
      ["machine learning", "combined"],
      ["neural networks", "combined"]
    ],
    "tfidf_keywords": [...],
    "rake_keywords": [...]
  }
}
```

### CSV Output Format
```csv
Rank,Keyword,Score,Method
1,machine learning,,combined
2,neural networks,,combined
```

## 🛠️ Troubleshooting Commands

### Check if application is working
```bash
python main.py --help
```

### Test with your PDF
```bash
python main.py One_token_to_fool_LLM.pdf --keywords 10
```

### Verify environment
```bash
# Check if in virtual environment (should show (keyword_extractor))
which python

# List installed packages
pip list
```

### Re-download NLTK data if needed
```bash
python -c "import nltk; nltk.download('all')"
```

## 📈 Performance Tips

### For Large PDFs
```bash
# Start with fewer keywords for faster processing
python main.py large_paper.pdf --keywords 20

# Then increase if needed
python main.py large_paper.pdf --keywords 50 --save --format json
```

### For Many Files
```bash
# Process in batches if you have many files
python main.py --directory ./batch1 --keywords 25 --save
python main.py --directory ./batch2 --keywords 25 --save
```

## 🎯 Real-World Examples

### Example 1: Analyzing Your Research Paper
```bash
python main.py "One_token_to_fool_LLM.pdf" --keywords 30 --save --format json
```

### Example 2: Literature Review Workflow
```bash
# Step 1: Quick screening
python main.py --directory ./candidate_papers --keywords 15

# Step 2: Detailed analysis of selected papers  
python main.py selected_paper.pdf --keywords 50 --save --format json
```

### Example 3: Building Keyword Database
```bash
# Extract comprehensive keywords for database
python main.py --directory ./all_papers --keywords 40 --save --format csv
```

## 📝 Quick Reference Card

| Command | Short | Description |
|---------|-------|-------------|
| `--keywords N` | `-k N` | Extract N keywords |
| `--save` | `-s` | Save results to file |
| `--format json` | `-f json` | Output in JSON format |
| `--format csv` | `-f csv` | Output in CSV format |
| `--directory PATH` | `-d PATH` | Process all PDFs in directory |
| `--help` | `-h` | Show help message |
| `--version` | | Show version information |

## 🏆 Recommended Workflows

### Quick Paper Assessment
```bash
python main.py paper.pdf --keywords 20
```

### Comprehensive Analysis
```bash
python main.py paper.pdf --keywords 50 --save --format json
```

### Research Database Building
```bash
python main.py --directory ./papers --keywords 35 --save --format csv
```

---

**💡 Pro Tip**: Start with fewer keywords (15-25) for quick analysis, then increase to 50+ for comprehensive extraction. Always save important results using `--save --format json` for future reference!
