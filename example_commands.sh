#!/bin/bash
# AI Paper Keyword Extractor - Example Commands
# Make this file executable with: chmod +x example_commands.sh

echo "AI Paper Keyword Extractor - Example Commands"
echo "=============================================="
echo ""

# Activate virtual environment first
echo "🔧 Activating virtual environment..."
source keyword_extractor/bin/activate

echo ""
echo "📚 BASIC USAGE EXAMPLES:"
echo "------------------------"

echo ""
echo "1. Basic extraction (25 keywords - default):"
echo "   python main.py One_token_to_fool_LLM.pdf"

echo ""
echo "2. Extract more keywords:"
echo "   python main.py One_token_to_fool_LLM.pdf --keywords 50"

echo ""
echo "3. Save results to file:"
echo "   python main.py One_token_to_fool_LLM.pdf --save"

echo ""
echo "4. Save as JSON for detailed analysis:"
echo "   python main.py One_token_to_fool_LLM.pdf --keywords 40 --save --format json"

echo ""
echo "5. Save as CSV for spreadsheet use:"
echo "   python main.py One_token_to_fool_LLM.pdf --keywords 30 --save --format csv"

echo ""
echo "📁 BATCH PROCESSING EXAMPLES:"
echo "-----------------------------"

echo ""
echo "6. Process all PDFs in a directory:"
echo "   python main.py --directory ./papers"

echo ""
echo "7. Batch process with custom settings:"
echo "   python main.py --directory ./papers --keywords 35 --save --format json"

echo ""
echo "🎯 RESEARCH WORKFLOW EXAMPLES:"
echo "------------------------------"

echo ""
echo "8. Quick paper screening:"
echo "   python main.py paper.pdf --keywords 15"

echo ""
echo "9. Comprehensive analysis:"
echo "   python main.py paper.pdf --keywords 75 --save --format json"

echo ""
echo "10. Literature review workflow:"
echo "    python main.py --directory ./literature --keywords 30 --save"

echo ""
echo "⚡ QUICK COMMANDS (using short flags):"
echo "-------------------------------------"

echo ""
echo "11. Quick with short flags:"
echo "    python main.py One_token_to_fool_LLM.pdf -k 40 -s -f json"

echo ""
echo "12. Directory batch with short flags:"
echo "    python main.py -d ./papers -k 25 -s"

echo ""
echo "🔍 TESTING AND TROUBLESHOOTING:"
echo "-------------------------------"

echo ""
echo "13. Show help:"
echo "    python main.py --help"

echo ""
echo "14. Test with sample PDF:"
echo "    python main.py One_token_to_fool_LLM.pdf -k 10"

echo ""
echo "15. Check version:"
echo "    python main.py --version"

echo ""
echo "===========================================" 
echo "💡 To run any of these commands, copy and paste them into your terminal"
echo "💡 Remember to activate the virtual environment first!"
echo "💡 Replace 'One_token_to_fool_LLM.pdf' with your actual PDF filename"
echo "==========================================="
