#!/usr/bin/env python3
"""
AI Paper Keyword Extractor
Main application for extracting keywords from research papers in PDF format
"""

import argparse
import sys
import os
import json
from datetime import datetime
from pathlib import Path

# Import our modules
from pdf_reader import PDFReader
from text_processor import TextProcessor
from keyword_extractor import KeywordExtractor


class PaperKeywordExtractor:
    """Main class for the AI Paper Keyword Extractor application"""
    
    def __init__(self):
        """Initialize the extractor with required components"""
        self.pdf_reader = PDFReader()
        self.text_processor = TextProcessor()
        self.keyword_extractor = KeywordExtractor()
    
    def extract_keywords_from_pdf(self, pdf_path: str, num_keywords: int = 15, 
                                 save_output: bool = False, output_format: str = 'text') -> dict:
        """
        Extract keywords from a PDF file
        
        Args:
            pdf_path (str): Path to the PDF file
            num_keywords (int): Number of keywords to extract
            save_output (bool): Whether to save output to file
            output_format (str): Output format ('text', 'json', 'csv')
            
        Returns:
            dict: Extraction results
        """
        try:
            print(f"Processing PDF: {pdf_path}")
            
            # Step 1: Extract text from PDF
            print("Step 1: Extracting text from PDF...")
            raw_text = self.pdf_reader.extract_text(pdf_path)
            print(f"Extracted {len(raw_text)} characters")
            
            # Get PDF info
            pdf_info = self.pdf_reader.get_pdf_info(pdf_path)
            
            # Step 2: Preprocess text
            print("Step 2: Preprocessing text...")
            processed_data = self.text_processor.preprocess_for_keywords(raw_text)
            print(f"Processed to {processed_data['word_count']} words "
                  f"({processed_data['unique_words']} unique)")
            
            # Step 3: Extract keywords
            print("Step 3: Extracting keywords...")
            keyword_summary = self.keyword_extractor.get_keyword_summary(
                raw_text, processed_data, top_k=num_keywords
            )
            
            # Prepare results
            results = {
                'pdf_info': pdf_info,
                'text_stats': {
                    'total_characters': len(raw_text),
                    'word_count': processed_data['word_count'],
                    'unique_words': processed_data['unique_words']
                },
                'keyword_summary': keyword_summary,
                'extraction_timestamp': datetime.now().isoformat(),
                'pdf_path': pdf_path
            }
            
            # Display results
            self._display_results(results)
            
            # Save output if requested
            if save_output:
                self._save_results(results, pdf_path, output_format)
            
            return results
            
        except Exception as e:
            error_msg = f"Error processing PDF: {str(e)}"
            print(f"ERROR: {error_msg}")
            return {'error': error_msg}
    
    def process_multiple_pdfs(self, pdf_directory: str, num_keywords: int = 15,
                            save_output: bool = False, output_format: str = 'text'):
        """
        Process multiple PDF files in a directory
        
        Args:
            pdf_directory (str): Directory containing PDF files
            num_keywords (int): Number of keywords to extract per PDF
            save_output (bool): Whether to save outputs
            output_format (str): Output format
        """
        if not os.path.exists(pdf_directory):
            print(f"ERROR: Directory not found: {pdf_directory}")
            return
        
        pdf_files = [f for f in os.listdir(pdf_directory) if f.lower().endswith('.pdf')]
        
        if not pdf_files:
            print(f"No PDF files found in: {pdf_directory}")
            return
        
        print(f"Found {len(pdf_files)} PDF files to process")
        
        all_results = []
        for i, pdf_file in enumerate(pdf_files, 1):
            pdf_path = os.path.join(pdf_directory, pdf_file)
            print(f"\n{'='*60}")
            print(f"Processing {i}/{len(pdf_files)}: {pdf_file}")
            print(f"{'='*60}")
            
            result = self.extract_keywords_from_pdf(
                pdf_path, num_keywords, save_output, output_format
            )
            all_results.append(result)
        
        # Save combined results
        if save_output:
            self._save_batch_results(all_results, pdf_directory, output_format)
    
    def _display_results(self, results: dict):
        """Display extraction results in a formatted way"""
        print(f"\n{'='*60}")
        print("KEYWORD EXTRACTION RESULTS")
        print(f"{'='*60}")
        
        # PDF Info
        if 'pdf_info' in results and 'error' not in results['pdf_info']:
            info = results['pdf_info']
            print(f"\nPDF Information:")
            print(f"  Title: {info.get('title', 'Unknown')}")
            print(f"  Author: {info.get('author', 'Unknown')}")
            print(f"  Pages: {info.get('num_pages', 'Unknown')}")
        
        # Text Stats
        if 'text_stats' in results:
            stats = results['text_stats']
            print(f"\nText Statistics:")
            print(f"  Total characters: {stats['total_characters']:,}")
            print(f"  Word count: {stats['word_count']:,}")
            print(f"  Unique words: {stats['unique_words']:,}")
        
        # Keywords
        if 'keyword_summary' in results:
            summary = results['keyword_summary']
            
            print(f"\n🔑 RECOMMENDED KEYWORDS:")
            print(f"{'-'*40}")
            for i, (keyword, source) in enumerate(summary['recommended_keywords'], 1):
                print(f"{i:2d}. {keyword:<25} [{source}]")
            
            print(f"\n📊 TF-IDF KEYWORDS:")
            print(f"{'-'*40}")
            for i, (keyword, score) in enumerate(summary['tfidf_keywords'][:20], 1):
                print(f"{i:2d}. {keyword:<25} ({score:.3f})")
            
            print(f"\n🔍 RAKE KEYWORDS:")
            print(f"{'-'*40}")
            for i, (keyword, score) in enumerate(summary['rake_keywords'][:20], 1):
                print(f"{i:2d}. {keyword:<25} ({score:.3f})")
            
            if summary['noun_phrase_keywords']:
                print(f"\n📝 KEY PHRASES:")
                print(f"{'-'*40}")
                for i, (phrase, count) in enumerate(summary['noun_phrase_keywords'][:10], 1):
                    print(f"{i:2d}. {phrase:<30} (x{count})")
    
    def _save_results(self, results: dict, pdf_path: str, output_format: str):
        """Save results to file"""
        base_name = Path(pdf_path).stem
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if output_format == 'json':
            output_file = f"{base_name}_keywords_{timestamp}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
        
        elif output_format == 'csv':
            output_file = f"{base_name}_keywords_{timestamp}.csv"
            self._save_as_csv(results, output_file)
        
        else:  # text format
            output_file = f"{base_name}_keywords_{timestamp}.txt"
            self._save_as_text(results, output_file)
        
        print(f"\nResults saved to: {output_file}")
    
    def _save_as_text(self, results: dict, filename: str):
        """Save results as text file"""
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("AI PAPER KEYWORD EXTRACTION RESULTS\n")
            f.write("="*50 + "\n\n")
            
            if 'pdf_path' in results:
                f.write(f"PDF File: {results['pdf_path']}\n")
            f.write(f"Extraction Time: {results.get('extraction_timestamp', 'Unknown')}\n\n")
            
            # Write recommended keywords
            if 'keyword_summary' in results:
                summary = results['keyword_summary']
                
                f.write("RECOMMENDED KEYWORDS:\n")
                f.write("-" * 30 + "\n")
                for i, (keyword, source) in enumerate(summary['recommended_keywords'], 1):
                    f.write(f"{i:2d}. {keyword} [{source}]\n")
                
                f.write(f"\nTF-IDF KEYWORDS:\n")
                f.write("-" * 30 + "\n")
                for i, (keyword, score) in enumerate(summary['tfidf_keywords'], 1):
                    f.write(f"{i:2d}. {keyword} ({score:.3f})\n")
                
                f.write(f"\nRAKE KEYWORDS:\n")
                f.write("-" * 30 + "\n")
                for i, (keyword, score) in enumerate(summary['rake_keywords'], 1):
                    f.write(f"{i:2d}. {keyword} ({score:.3f})\n")
    
    def _save_as_csv(self, results: dict, filename: str):
        """Save results as CSV file"""
        import csv
        
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Rank', 'Keyword', 'Score', 'Method'])
            
            if 'keyword_summary' in results:
                summary = results['keyword_summary']
                
                # Write recommended keywords
                for i, (keyword, source) in enumerate(summary['recommended_keywords'], 1):
                    writer.writerow([i, keyword, '', source])
    
    def _save_batch_results(self, all_results: list, directory: str, output_format: str):
        """Save combined results from batch processing"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"batch_keywords_{timestamp}.{output_format}"
        
        if output_format == 'json':
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(all_results, f, indent=2, ensure_ascii=False)
        else:
            # For text/csv, create summary
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write("BATCH KEYWORD EXTRACTION SUMMARY\n")
                f.write("="*50 + "\n\n")
                
                for i, result in enumerate(all_results, 1):
                    if 'error' not in result:
                        f.write(f"{i}. {Path(result.get('pdf_path', 'Unknown')).name}\n")
                        if 'keyword_summary' in result:
                            keywords = result['keyword_summary']['recommended_keywords'][:5]
                            for keyword, source in keywords:
                                f.write(f"   - {keyword}\n")
                        f.write("\n")
        
        print(f"\nBatch results saved to: {output_file}")


def main():
    """Main function with CLI interface"""
    parser = argparse.ArgumentParser(
        description="AI Paper Keyword Extractor - Extract keywords from research papers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py paper.pdf
  python main.py paper.pdf --keywords 20 --save --format json
  python main.py --directory ./papers --keywords 15 --save
        """
    )
    
    # Main arguments
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('pdf_file', nargs='?', help='Path to PDF file to process')
    group.add_argument('--directory', '-d', help='Directory containing PDF files to process')
    
    # Optional arguments
    parser.add_argument('--keywords', '-k', type=int, default=25,
                       help='Number of keywords to extract (default: 25)')
    parser.add_argument('--save', '-s', action='store_true',
                       help='Save results to file')
    parser.add_argument('--format', '-f', choices=['text', 'json', 'csv'], default='text',
                       help='Output format (default: text)')
    parser.add_argument('--version', action='version', version='AI Paper Keyword Extractor 1.0.0')
    
    args = parser.parse_args()
    
    # Initialize extractor
    extractor = PaperKeywordExtractor()
    
    try:
        if args.pdf_file:
            # Process single PDF
            if not os.path.exists(args.pdf_file):
                print(f"ERROR: File not found: {args.pdf_file}")
                sys.exit(1)
            
            extractor.extract_keywords_from_pdf(
                args.pdf_file, 
                args.keywords, 
                args.save, 
                args.format
            )
        
        elif args.directory:
            # Process directory of PDFs
            extractor.process_multiple_pdfs(
                args.directory,
                args.keywords,
                args.save,
                args.format
            )
    
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        sys.exit(0)
    except Exception as e:
        print(f"ERROR: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
