"""
PDF Reader Module for AI Paper Keyword Extractor
Extracts text content from PDF files using PyPDF2

Copyright (c) 2024 Sreeram Lagisetty. All rights reserved.
This project is proprietary software. Unauthorized copying, distribution, or use is strictly prohibited.
"""

import PyPDF2
import os
# str is a built-in type, no import needed


class PDFReader:
    """Class to handle PDF text extraction"""
    
    def __init__(self):
        self.supported_formats = ['.pdf']
    
    def is_supported_file(self, file_path: str) -> bool:
        """Check if the file format is supported"""
        _, ext = os.path.splitext(file_path.lower())
        return ext in self.supported_formats
    
    def extract_text(self, pdf_path: str) -> str:
        """
        Extract text from PDF file
        
        Args:
            pdf_path (str): Path to the PDF file
            
        Returns:
            str: Extracted text content
            
        Raises:
            FileNotFoundError: If PDF file doesn't exist
            Exception: For other PDF reading errors
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        if not self.is_supported_file(pdf_path):
            raise ValueError(f"Unsupported file format. Supported formats: {self.supported_formats}")
        
        try:
            text_content = ""
            
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                # Extract text from each page
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    text_content += page.extract_text() + "\n"
            
            if not text_content.strip():
                raise ValueError("No text content found in PDF. The PDF might be image-based or corrupted.")
            
            return text_content
            
        except Exception as e:
            raise Exception(f"Error reading PDF file: {str(e)}")
    
    def get_pdf_info(self, pdf_path: str) -> dict:
        """
        Get basic information about the PDF
        
        Args:
            pdf_path (str): Path to the PDF file
            
        Returns:
            dict: PDF metadata information
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                info = {
                    'num_pages': len(pdf_reader.pages),
                    'title': pdf_reader.metadata.title if pdf_reader.metadata and pdf_reader.metadata.title else 'Unknown',
                    'author': pdf_reader.metadata.author if pdf_reader.metadata and pdf_reader.metadata.author else 'Unknown',
                    'subject': pdf_reader.metadata.subject if pdf_reader.metadata and pdf_reader.metadata.subject else 'Unknown',
                    'creator': pdf_reader.metadata.creator if pdf_reader.metadata and pdf_reader.metadata.creator else 'Unknown'
                }
                
                return info
                
        except Exception as e:
            return {'error': f"Error reading PDF metadata: {str(e)}"}


if __name__ == "__main__":
    # Test the PDF reader
    reader = PDFReader()
    
    # Example usage
    test_pdf = "sample_paper.pdf"
    if os.path.exists(test_pdf):
        try:
            text = reader.extract_text(test_pdf)
            info = reader.get_pdf_info(test_pdf)
            
            print(f"PDF Info: {info}")
            print(f"Text length: {len(text)} characters")
            print(f"First 500 characters:\n{text[:500]}...")
            
        except Exception as e:
            print(f"Error: {e}")
    else:
        print("No test PDF file found. Place a PDF file named 'sample_paper.pdf' in the current directory to test.")
