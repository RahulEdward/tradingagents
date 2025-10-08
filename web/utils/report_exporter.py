#!/usr/bin/env python3
"""
Report Export Tool
Supports exporting analysis results to multiple formats
"""

import streamlit as st
import json
import os
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import tempfile
import base64

# Import logging module
from tradingagents.utils.logging_manager import get_logger
logger = get_logger('web')

# Import MongoDB report manager
try:
    from web.utils.mongodb_report_manager import mongodb_report_manager
    MONGODB_REPORT_AVAILABLE = True
except ImportError:
    MONGODB_REPORT_AVAILABLE = False
    mongodb_report_manager = None

# Configure logging - ensure output to stdout for Docker logs visibility
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Output to stdout
    ]
)
logger = logging.getLogger(__name__)

# Import Docker adapter
try:
    from .docker_pdf_adapter import (
        is_docker_environment,
        get_docker_pdf_extra_args,
        setup_xvfb_display,
        get_docker_status_info
    )
    DOCKER_ADAPTER_AVAILABLE = True
except ImportError:
    DOCKER_ADAPTER_AVAILABLE = False
    logger.warning(f"⚠️ Docker adapter not available")

# Import export-related libraries
try:
    import markdown
    import re
    import tempfile
    import os
    from pathlib import Path

    # Import pypandoc (for markdown to docx and pdf conversion)
    import pypandoc

    # Check if pandoc is available, try to download if not available
    try:
        pypandoc.get_pandoc_version()
        PANDOC_AVAILABLE = True
    except OSError:
        logger.warning(f"⚠️ Pandoc not found, attempting automatic download...")
        try:
            pypandoc.download_pandoc()
            PANDOC_AVAILABLE = True
            logger.info(f"✅ Pandoc download successful!")
        except Exception as download_error:
            logger.error(f"❌ Pandoc download failed: {download_error}")
            PANDOC_AVAILABLE = False

    EXPORT_AVAILABLE = True

except ImportError as e:
    EXPORT_AVAILABLE = False
    PANDOC_AVAILABLE = False
    logger.info(f"Export functionality dependencies missing: {e}")
    logger.info(f"Please install: pip install pypandoc markdown")


class ReportExporter:
    """Report Exporter"""

    def __init__(self):
        self.export_available = EXPORT_AVAILABLE
        self.pandoc_available = PANDOC_AVAILABLE
        self.is_docker = DOCKER_ADAPTER_AVAILABLE and is_docker_environment()

        # Log initialization status
        logger.info(f"📋 ReportExporter initialization:")
        logger.info(f"  - export_available: {self.export_available}")
        logger.info(f"  - pandoc_available: {self.pandoc_available}")
        logger.info(f"  - is_docker: {self.is_docker}")
        logger.info(f"  - docker_adapter_available: {DOCKER_ADAPTER_AVAILABLE}")

        # Docker environment initialization
        if self.is_docker:
            logger.info("🐳 Docker environment detected, initializing PDF support...")
            logger.info(f"🐳 Docker environment detected, initializing PDF support...")
            setup_xvfb_display()
    
    def _clean_text_for_markdown(self, text: str) -> str:
        """Clean characters in text that may cause YAML parsing issues"""
        if not text:
            return "N/A"

        # Convert to string and clean special characters
        text = str(text)

        # Remove characters that may cause YAML parsing issues
        text = text.replace('&', '&amp;')  # HTML escape
        text = text.replace('<', '&lt;')
        text = text.replace('>', '&gt;')
        text = text.replace('"', '&quot;')
        text = text.replace("'", '&#39;')

        # Remove possible YAML special characters
        text = text.replace('---', '—')  # Replace three hyphens
        text = text.replace('...', '…')  # Replace three dots

        return text

    def _clean_markdown_for_pandoc(self, content: str) -> str:
        """Clean Markdown content to avoid pandoc YAML parsing issues"""
        if not content:
            return ""

        # Ensure content doesn't start with characters that might be mistaken for YAML
        content = content.strip()

        # If first line looks like YAML delimiter, add empty line
        lines = content.split('\n')
        if lines and (lines[0].startswith('---') or lines[0].startswith('...')):
            content = '\n' + content

        # Replace character sequences that may cause YAML parsing issues, but protect table separators
        # First protect table separators
        content = content.replace('|------|------|', '|TABLESEP|TABLESEP|')
        content = content.replace('|------|', '|TABLESEP|')

        # Then replace other triple hyphens
        content = content.replace('---', '—')  # Replace three hyphens
        content = content.replace('...', '…')  # Replace three dots

        # Restore table separators
        content = content.replace('|TABLESEP|TABLESEP|', '|------|------|')
        content = content.replace('|TABLESEP|', '|------|')

        # Clean special quotes
        content = content.replace('"', '"')  # Left double quote
        content = content.replace('"', '"')  # Right double quote
        content = content.replace(''', "'")  # Left single quote
        content = content.replace(''', "'")  # Right single quote

        # Ensure content starts with standard Markdown heading
        if not content.startswith('#'):
            content = '# Analysis Report\n\n' + content

        return content

    def generate_markdown_report(self, results: Dict[str, Any]) -> str:
        """Generate Markdown format report"""

        stock_symbol = self._clean_text_for_markdown(results.get('stock_symbol', 'N/A'))
        decision = results.get('decision', {})
        state = results.get('state', {})
        is_demo = results.get('is_demo', False)
        
        # Generate timestamp
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Clean key data
        action = self._clean_text_for_markdown(decision.get('action', 'N/A')).upper()
        target_price = self._clean_text_for_markdown(decision.get('target_price', 'N/A'))
        reasoning = self._clean_text_for_markdown(decision.get('reasoning', 'No analysis reasoning available'))

        # Build Markdown content
        md_content = f"""# {stock_symbol} Stock Analysis Report

**Generation Time**: {timestamp}
**Analysis Status**: {'Demo Mode' if is_demo else 'Formal Analysis'}

## 🎯 Investment Decision Summary

| Metric | Value |
|------|------|
| **Investment Recommendation** | {action} |
| **Confidence** | {decision.get('confidence', 0):.1%} |
| **Risk Score** | {decision.get('risk_score', 0):.1%} |
| **Target Price** | {target_price} |

### Analysis Reasoning
{reasoning}

---

## 📋 Analysis Configuration Information

- **LLM Provider**: {results.get('llm_provider', 'N/A')}
- **AI Model**: {results.get('llm_model', 'N/A')}
- **Number of Analysts**: {len(results.get('analysts', []))} analysts
- **Research Depth**: {results.get('research_depth', 'N/A')}

### Participating Analysts
{', '.join(results.get('analysts', []))}

---

## 📊 Detailed Analysis Report

"""
        
        # Add content from each analysis module - maintain complete structure consistent with CLI
        analysis_modules = [
            ('market_report', '📈 Market Technical Analysis', 'Technical indicators, price trends, support and resistance analysis'),
            ('fundamentals_report', '💰 Fundamental Analysis', 'Financial data, valuation levels, profitability analysis'),
            ('sentiment_report', '💭 Market Sentiment Analysis', 'Investor sentiment, social media sentiment indicators'),
            ('news_report', '📰 News Event Analysis', 'Related news events, market dynamics impact analysis'),
            ('risk_assessment', '⚠️ Risk Assessment', 'Risk factor identification, risk level assessment'),
            ('investment_plan', '📋 Investment Recommendations', 'Specific investment strategies, position management recommendations')
        ]
        
        for key, title, description in analysis_modules:
            md_content += f"\n### {title}\n\n"
            md_content += f"*{description}*\n\n"
            
            if key in state and state[key]:
                content = state[key]
                if isinstance(content, str):
                    md_content += f"{content}\n\n"
                elif isinstance(content, dict):
                    for sub_key, sub_value in content.items():
                        md_content += f"#### {sub_key.replace('_', ' ').title()}\n\n"
                        md_content += f"{sub_value}\n\n"
                else:
                    md_content += f"{content}\n\n"
            else:
                md_content += "No data available\n\n"

        # Add team decision report section - maintain consistency with CLI
        md_content = self._add_team_decision_reports(md_content, state)

        # Add risk disclaimer
        md_content += f"""
---

## ⚠️ Important Risk Disclaimer

**Investment Risk Warning**:
- **For Reference Only**: This analysis result is for reference only and does not constitute investment advice
- **Investment Risk**: Stock investment involves risks and may result in principal loss
- **Rational Decision**: Please make rational investment decisions by combining multiple sources of information
- **Professional Consultation**: Major investment decisions should consult professional financial advisors
- **Self-Responsibility**: Investment decisions and their consequences are borne by investors themselves

---
*Report Generation Time: {timestamp}*
"""
        
        return md_content

    def _add_team_decision_reports(self, md_content: str, state: Dict[str, Any]) -> str:
        """Add team decision report section, maintain consistency with CLI"""

        # II. Research Team Decision Report
        if 'investment_debate_state' in state and state['investment_debate_state']:
            md_content += "\n---\n\n## 🔬 Research Team Decision\n\n"
            md_content += "*Bull/Bear analyst debate analysis, research manager comprehensive decision*\n\n"

            debate_state = state['investment_debate_state']

            # Bull analyst analysis
            if debate_state.get('bull_history'):
                md_content += "### 📈 Bull Analyst Analysis\n\n"
                md_content += f"{self._clean_text_for_markdown(debate_state['bull_history'])}\n\n"

            # Bear analyst analysis
            if debate_state.get('bear_history'):
                md_content += "### 📉 Bear Analyst Analysis\n\n"
                md_content += f"{self._clean_text_for_markdown(debate_state['bear_history'])}\n\n"

            # Research manager decision
            if debate_state.get('judge_decision'):
                md_content += "### 🎯 Research Manager Comprehensive Decision\n\n"
                md_content += f"{self._clean_text_for_markdown(debate_state['judge_decision'])}\n\n"

        # III. Trading Team Plan
        if 'trader_investment_plan' in state and state['trader_investment_plan']:
            md_content += "\n---\n\n## 💼 Trading Team Plan\n\n"
            md_content += "*Specific trading execution plan developed by professional traders*\n\n"
            md_content += f"{self._clean_text_for_markdown(state['trader_investment_plan'])}\n\n"

        # IV. Risk Management Team Decision
        if 'risk_debate_state' in state and state['risk_debate_state']:
            md_content += "\n---\n\n## ⚖️ Risk Management Team Decision\n\n"
            md_content += "*Aggressive/Conservative/Neutral analyst risk assessment, portfolio manager final decision*\n\n"

            risk_state = state['risk_debate_state']

            # Aggressive analyst
            if risk_state.get('risky_history'):
                md_content += "### 🚀 Aggressive Analyst Assessment\n\n"
                md_content += f"{self._clean_text_for_markdown(risk_state['risky_history'])}\n\n"

            # Conservative analyst
            if risk_state.get('safe_history'):
                md_content += "### 🛡️ Conservative Analyst Assessment\n\n"
                md_content += f"{self._clean_text_for_markdown(risk_state['safe_history'])}\n\n"

            # Neutral analyst
            if risk_state.get('neutral_history'):
                md_content += "### ⚖️ Neutral Analyst Assessment\n\n"
                md_content += f"{self._clean_text_for_markdown(risk_state['neutral_history'])}\n\n"

            # Portfolio manager decision
            if risk_state.get('judge_decision'):
                md_content += "### 🎯 Portfolio Manager Final Decision\n\n"
                md_content += f"{self._clean_text_for_markdown(risk_state['judge_decision'])}\n\n"

        # V. Final Trading Decision
        if 'final_trade_decision' in state and state['final_trade_decision']:
            md_content += "\n---\n\n## 🎯 Final Trading Decision\n\n"
            md_content += "*Final investment decision after comprehensive analysis from all teams*\n\n"
            md_content += f"{self._clean_text_for_markdown(state['final_trade_decision'])}\n\n"

        return md_content

    def _format_team_decision_content(self, content: Dict[str, Any], module_key: str) -> str:
        """Format team decision content"""
        formatted_content = ""

        if module_key == 'investment_debate_state':
            # Research team decision formatting
            if content.get('bull_history'):
                formatted_content += "## 📈 Bull Analyst Analysis\n\n"
                formatted_content += f"{content['bull_history']}\n\n"

            if content.get('bear_history'):
                formatted_content += "## 📉 Bear Analyst Analysis\n\n"
                formatted_content += f"{content['bear_history']}\n\n"

            if content.get('judge_decision'):
                formatted_content += "## 🎯 Research Manager Comprehensive Decision\n\n"
                formatted_content += f"{content['judge_decision']}\n\n"

        elif module_key == 'risk_debate_state':
            # Risk management team decision formatting
            if content.get('risky_history'):
                formatted_content += "## 🚀 Aggressive Analyst Assessment\n\n"
                formatted_content += f"{content['risky_history']}\n\n"

            if content.get('safe_history'):
                formatted_content += "## 🛡️ Conservative Analyst Assessment\n\n"
                formatted_content += f"{content['safe_history']}\n\n"

            if content.get('neutral_history'):
                formatted_content += "## ⚖️ Neutral Analyst Assessment\n\n"
                formatted_content += f"{content['neutral_history']}\n\n"

            if content.get('judge_decision'):
                formatted_content += "## 🎯 Portfolio Manager Final Decision\n\n"
                formatted_content += f"{content['judge_decision']}\n\n"

        return formatted_content

    def generate_docx_report(self, results: Dict[str, Any]) -> bytes:
        """Generate Word document format report"""

        logger.info("📄 Starting Word document generation...")

        if not self.pandoc_available:
            logger.error("❌ Pandoc not available")
            raise Exception("Pandoc not available, cannot generate Word document. Please install pandoc or use Markdown format export.")

        # First generate markdown content
        logger.info("📝 Generating Markdown content...")
        md_content = self.generate_markdown_report(results)
        logger.info(f"✅ Markdown content generation completed, length: {len(md_content)} characters")

        try:
            logger.info("📁 Creating temporary file for docx output...")
            # Create temporary file for docx output
            with tempfile.NamedTemporaryFile(suffix='.docx', delete=False) as tmp_file:
                output_file = tmp_file.name
            logger.info(f"📁 Temporary file path: {output_file}")

            # Use parameters to force disable YAML
            extra_args = ['--from=markdown-yaml_metadata_block']  # Disable YAML parsing
            logger.info(f"🔧 pypandoc parameters: {extra_args} (disable YAML parsing)")

            logger.info("🔄 Converting markdown to docx using pypandoc...")

            # Debug: save actual Markdown content
            debug_file = '/app/debug_markdown.md'
            try:
                with open(debug_file, 'w', encoding='utf-8') as f:
                    f.write(md_content)
                logger.info(f"🔍 Actual Markdown content saved to: {debug_file}")
                logger.info(f"📊 Content length: {len(md_content)} characters")

                # Display first few lines of content
                lines = md_content.split('\n')[:5]
                logger.info("🔍 First 5 lines of content:")
                for i, line in enumerate(lines, 1):
                    logger.info(f"  {i}: {repr(line)}")
            except Exception as e:
                logger.error(f"Failed to save debug file: {e}")

            # Clean content to avoid YAML parsing issues
            cleaned_content = self._clean_markdown_for_pandoc(md_content)
            logger.info(f"🧹 Content cleaning completed, cleaned length: {len(cleaned_content)} characters")

            # Use tested successful parameters for conversion
            pypandoc.convert_text(
                cleaned_content,
                'docx',
                format='markdown',  # Basic markdown format
                outputfile=output_file,
                extra_args=extra_args
            )
            logger.info("✅ pypandoc conversion completed")

            logger.info("📖 Reading generated docx file...")
            # Read generated docx file
            with open(output_file, 'rb') as f:
                docx_content = f.read()
            logger.info(f"✅ File reading completed, size: {len(docx_content)} bytes")

            logger.info("🗑️ Cleaning up temporary files...")
            # Clean up temporary files
            os.unlink(output_file)
            logger.info("✅ Temporary file cleanup completed")

            return docx_content
        except Exception as e:
            logger.error(f"❌ Word document generation failed: {e}", exc_info=True)
            raise Exception(f"Word document generation failed: {e}")
    
    
    def generate_pdf_report(self, results: Dict[str, Any]) -> bytes:
        """Generate PDF format report"""

        logger.info("📊 Starting PDF document generation...")

        if not self.pandoc_available:
            logger.error("❌ Pandoc not available")
            raise Exception("Pandoc not available, cannot generate PDF document. Please install pandoc or use Markdown format export.")

        # First generate markdown content
        logger.info("📝 Generating Markdown content...")
        md_content = self.generate_markdown_report(results)
        logger.info(f"✅ Markdown content generation completed, length: {len(md_content)} characters")

        # Simplified PDF engine list, prioritizing most likely to succeed
        pdf_engines = [
            ('wkhtmltopdf', 'HTML to PDF engine, recommended installation'),
            ('weasyprint', 'Modern HTML to PDF engine'),
            (None, 'Use pandoc default engine')  # Don't specify engine, let pandoc choose
        ]

        last_error = None

        for engine_info in pdf_engines:
            engine, description = engine_info
            try:
                # Create temporary file for PDF output
                with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
                    output_file = tmp_file.name

                # Use parameters to disable YAML parsing (consistent with Word export)
                extra_args = ['--from=markdown-yaml_metadata_block']

                # If engine is specified, add engine parameter
                if engine:
                    extra_args.append(f'--pdf-engine={engine}')
                    logger.info(f"🔧 Using PDF engine: {engine}")
                else:
                    logger.info(f"🔧 Using default PDF engine")

                logger.info(f"🔧 PDF parameters: {extra_args}")

                # Clean content to avoid YAML parsing issues (consistent with Word export)
                cleaned_content = self._clean_markdown_for_pandoc(md_content)

                # Use pypandoc to convert markdown to PDF - disable YAML parsing
                pypandoc.convert_text(
                    cleaned_content,
                    'pdf',
                    format='markdown',  # Basic markdown format
                    outputfile=output_file,
                    extra_args=extra_args
                )

                # Check if file is generated and has content
                if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
                    # Read generated PDF file
                    with open(output_file, 'rb') as f:
                        pdf_content = f.read()

                    # Clean up temporary files
                    os.unlink(output_file)

                    logger.info(f"✅ PDF generation successful, using engine: {engine or 'default'}")
                    return pdf_content
                else:
                    raise Exception("PDF file generation failed or empty")

            except Exception as e:
                last_error = str(e)
                logger.error(f"PDF engine {engine or 'default'} failed: {e}")

                # Clean up possible existing temporary files
                try:
                    if 'output_file' in locals() and os.path.exists(output_file):
                        os.unlink(output_file)
                except:
                    pass

                continue

        # If all engines fail, provide detailed error information and solutions
        error_msg = f"""PDF generation failed, last error: {last_error}

Possible solutions:
1. Install wkhtmltopdf (recommended):
   Windows: choco install wkhtmltopdf
   macOS: brew install wkhtmltopdf
   Linux: sudo apt-get install wkhtmltopdf

2. Install LaTeX:
   Windows: choco install miktex
   macOS: brew install mactex
   Linux: sudo apt-get install texlive-full

3. Use Markdown or Word format export as alternative
"""
        raise Exception(error_msg)
    
    def export_report(self, results: Dict[str, Any], format_type: str) -> Optional[bytes]:
        """Export report to specified format"""

        logger.info(f"🚀 Starting report export: format={format_type}")
        logger.info(f"📊 Export status check:")
        logger.info(f"  - export_available: {self.export_available}")
        logger.info(f"  - pandoc_available: {self.pandoc_available}")
        logger.info(f"  - is_docker: {self.is_docker}")

        if not self.export_available:
            logger.error("❌ Export functionality not available")
            st.error("❌ Export functionality not available, please install necessary dependencies")
            return None

        try:
            logger.info(f"🔄 Starting {format_type} format report generation...")

            if format_type == 'markdown':
                logger.info("📝 Generating Markdown report...")
                content = self.generate_markdown_report(results)
                logger.info(f"✅ Markdown report generation successful, length: {len(content)} characters")
                return content.encode('utf-8')

            elif format_type == 'docx':
                logger.info("📄 Generating Word document...")
                if not self.pandoc_available:
                    logger.error("❌ pandoc not available, cannot generate Word document")
                    st.error("❌ pandoc not available, cannot generate Word document")
                    return None
                content = self.generate_docx_report(results)
                logger.info(f"✅ Word document generation successful, size: {len(content)} bytes")
                return content

            elif format_type == 'pdf':
                logger.info("📊 Generating PDF document...")
                if not self.pandoc_available:
                    logger.error("❌ pandoc not available, cannot generate PDF document")
                    st.error("❌ pandoc not available, cannot generate PDF document")
                    return None
                content = self.generate_pdf_report(results)
                logger.info(f"✅ PDF document generation successful, size: {len(content)} bytes")
                return content

            else:
                logger.error(f"❌ Unsupported export format: {format_type}")
                st.error(f"❌ Unsupported export format: {format_type}")
                return None

        except Exception as e:
            logger.error(f"❌ Export failed: {str(e)}", exc_info=True)
            st.error(f"❌ Export failed: {str(e)}")
            return None


# Create global exporter instance
report_exporter = ReportExporter()


def _format_team_decision_content(content: Dict[str, Any], module_key: str) -> str:
    """Format team decision content (standalone function version)"""
    formatted_content = ""

    if module_key == 'investment_debate_state':
        # Research team decision formatting
        if content.get('bull_history'):
            formatted_content += "## 📈 Bull Analyst Analysis\n\n"
            formatted_content += f"{content['bull_history']}\n\n"

        if content.get('bear_history'):
            formatted_content += "## 📉 Bear Analyst Analysis\n\n"
            formatted_content += f"{content['bear_history']}\n\n"

        if content.get('judge_decision'):
            formatted_content += "## 🎯 Research Manager Comprehensive Decision\n\n"
            formatted_content += f"{content['judge_decision']}\n\n"

    elif module_key == 'risk_debate_state':
        # Risk management team decision formatting
        if content.get('risky_history'):
            formatted_content += "## 🚀 Aggressive Analyst Assessment\n\n"
            formatted_content += f"{content['risky_history']}\n\n"

        if content.get('safe_history'):
            formatted_content += "## 🛡️ Conservative Analyst Assessment\n\n"
            formatted_content += f"{content['safe_history']}\n\n"

        if content.get('neutral_history'):
            formatted_content += "## ⚖️ Neutral Analyst Assessment\n\n"
            formatted_content += f"{content['neutral_history']}\n\n"

        if content.get('judge_decision'):
            formatted_content += "## 🎯 Portfolio Manager Final Decision\n\n"
            formatted_content += f"{content['judge_decision']}\n\n"

    return formatted_content


def save_modular_reports_to_results_dir(results: Dict[str, Any], stock_symbol: str) -> Dict[str, str]:
    """Save modular reports to results directory (CLI version format)"""
    try:
        import os
        from pathlib import Path

        # Get project root directory
        current_file = Path(__file__)
        project_root = current_file.parent.parent.parent

        # Get results directory configuration
        results_dir_env = os.getenv("TRADINGAGENTS_RESULTS_DIR")
        if results_dir_env:
            if not os.path.isabs(results_dir_env):
                results_dir = project_root / results_dir_env
            else:
                results_dir = Path(results_dir_env)
        else:
            results_dir = project_root / "results"

        # Create stock-specific directory
        analysis_date = datetime.now().strftime('%Y-%m-%d')
        stock_dir = results_dir / stock_symbol / analysis_date
        reports_dir = stock_dir / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)

        # Create message_tool.log file
        log_file = stock_dir / "message_tool.log"
        log_file.touch(exist_ok=True)

        state = results.get('state', {})
        saved_files = {}

        # Define report module mapping (consistent with CLI version)
        report_modules = {
            'market_report': {
                'filename': 'market_report.md',
                'title': f'{stock_symbol} Stock Technical Analysis Report',
                'state_key': 'market_report'
            },
            'sentiment_report': {
                'filename': 'sentiment_report.md',
                'title': f'{stock_symbol} Market Sentiment Analysis Report',
                'state_key': 'sentiment_report'
            },
            'news_report': {
                'filename': 'news_report.md',
                'title': f'{stock_symbol} News Event Analysis Report',
                'state_key': 'news_report'
            },
            'fundamentals_report': {
                'filename': 'fundamentals_report.md',
                'title': f'{stock_symbol} Fundamental Analysis Report',
                'state_key': 'fundamentals_report'
            },
            'investment_plan': {
                'filename': 'investment_plan.md',
                'title': f'{stock_symbol} Investment Decision Report',
                'state_key': 'investment_plan'
            },
            'trader_investment_plan': {
                'filename': 'trader_investment_plan.md',
                'title': f'{stock_symbol} Trading Plan Report',
                'state_key': 'trader_investment_plan'
            },
            'final_trade_decision': {
                'filename': 'final_trade_decision.md',
                'title': f'{stock_symbol} Final Investment Decision',
                'state_key': 'final_trade_decision'
            },
            # Add team decision report modules
            'investment_debate_state': {
                'filename': 'research_team_decision.md',
                'title': f'{stock_symbol} Research Team Decision Report',
                'state_key': 'investment_debate_state'
            },
            'risk_debate_state': {
                'filename': 'risk_management_decision.md',
                'title': f'{stock_symbol} Risk Management Team Decision Report',
                'state_key': 'risk_debate_state'
            }
        }

        # Generate report files for each module
        for module_key, module_info in report_modules.items():
            content = state.get(module_info['state_key'])

            if content:
                # Generate module report content
                if isinstance(content, str):
                    # Check if content already contains title to avoid duplication
                    if content.strip().startswith('#'):
                        report_content = content
                    else:
                        report_content = f"# {module_info['title']}\n\n{content}"
                elif isinstance(content, dict):
                    report_content = f"# {module_info['title']}\n\n"
                    # Special handling for team decision report dictionary structure
                    if module_key in ['investment_debate_state', 'risk_debate_state']:
                        report_content += _format_team_decision_content(content, module_key)
                    else:
                        for sub_key, sub_value in content.items():
                            report_content += f"## {sub_key.replace('_', ' ').title()}\n\n{sub_value}\n\n"
                else:
                    report_content = f"# {module_info['title']}\n\n{str(content)}"

                # Save file
                file_path = reports_dir / module_info['filename']
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(report_content)

                saved_files[module_key] = str(file_path)
                logger.info(f"✅ Saved module report: {file_path}")

        # If there is decision information, also save the final decision report
        decision = results.get('decision', {})
        if decision:
            decision_content = f"# {stock_symbol} Final Investment Decision\n\n"

            if isinstance(decision, dict):
                decision_content += f"## Investment Recommendation\n\n"
                decision_content += f"**Action**: {decision.get('action', 'N/A')}\n\n"
                decision_content += f"**Confidence**: {decision.get('confidence', 0):.1%}\n\n"
                decision_content += f"**Risk Score**: {decision.get('risk_score', 0):.1%}\n\n"
                decision_content += f"**Target Price**: {decision.get('target_price', 'N/A')}\n\n"
                decision_content += f"## Analysis Reasoning\n\n{decision.get('reasoning', 'No analysis reasoning available')}\n\n"
            else:
                decision_content += f"{str(decision)}\n\n"

            decision_file = reports_dir / "final_trade_decision.md"
            with open(decision_file, 'w', encoding='utf-8') as f:
                f.write(decision_content)

            saved_files['final_trade_decision'] = str(decision_file)
            logger.info(f"✅ Saved final decision: {decision_file}")

        # Save analysis metadata file, including research depth and other information
        metadata = {
            'stock_symbol': stock_symbol,
            'analysis_date': analysis_date,
            'timestamp': datetime.now().isoformat(),
            'research_depth': results.get('research_depth', 1),
            'analysts': results.get('analysts', []),
            'status': 'completed',
            'reports_count': len(saved_files),
            'report_types': list(saved_files.keys())
        }

        metadata_file = reports_dir.parent / "analysis_metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        logger.info(f"✅ Saved analysis metadata: {metadata_file}")
        logger.info(f"✅ Modular report saving completed, saved {len(saved_files)} files in total")
        logger.info(f"📁 Save directory: {os.path.normpath(str(reports_dir))}")

        # Also save to MongoDB
        logger.info(f"🔍 [MongoDB Debug] Starting MongoDB save process")
        logger.info(f"🔍 [MongoDB Debug] MONGODB_REPORT_AVAILABLE: {MONGODB_REPORT_AVAILABLE}")
        logger.info(f"🔍 [MongoDB Debug] mongodb_report_manager exists: {mongodb_report_manager is not None}")

        if MONGODB_REPORT_AVAILABLE and mongodb_report_manager:
            logger.info(f"🔍 [MongoDB Debug] MongoDB manager connection status: {mongodb_report_manager.connected}")
            try:
                # Collect all report content
                reports_content = {}

                logger.info(f"🔍 [MongoDB Debug] Starting to read {len(saved_files)} report files")
                # Read saved file content
                for module_key, file_path in saved_files.items():
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                            reports_content[module_key] = content
                            logger.info(f"🔍 [MongoDB Debug] Successfully read {module_key}: {len(content)} characters")
                    except Exception as e:
                        logger.warning(f"⚠️ Failed to read report file {file_path}: {e}")

                # Save to MongoDB
                if reports_content:
                    logger.info(f"🔍 [MongoDB Debug] Preparing to save to MongoDB, report count: {len(reports_content)}")
                    logger.info(f"🔍 [MongoDB Debug] Report types: {list(reports_content.keys())}")

                    success = mongodb_report_manager.save_analysis_report(
                        stock_symbol=stock_symbol,
                        analysis_results=results,
                        reports=reports_content
                    )

                    if success:
                        logger.info(f"✅ Analysis report has been saved to MongoDB as well")
                    else:
                        logger.warning(f"⚠️ MongoDB save failed, but file save succeeded")
                else:
                    logger.warning(f"⚠️ No report content available to save to MongoDB")

            except Exception as e:
                logger.error(f"❌ Error in MongoDB save process: {e}")
                import traceback
                logger.error(f"❌ Detailed MongoDB save error: {traceback.format_exc()}")
                # Does not affect successful file save return
        else:
            logger.warning(f"⚠️ MongoDB save skipped - AVAILABLE: {MONGODB_REPORT_AVAILABLE}, Manager: {mongodb_report_manager is not None}")

        return saved_files

    except Exception as e:
        logger.error(f"❌ Failed to save modular reports: {e}")
        import traceback
        logger.error(f"❌ Detailed error: {traceback.format_exc()}")
        return {}


def save_report_to_results_dir(content: bytes, filename: str, stock_symbol: str) -> str:
    """Save report to results directory"""
    try:
        import os
        from pathlib import Path

        # Get project root directory (Web application runs in web/ subdirectory)
        current_file = Path(__file__)
        project_root = current_file.parent.parent.parent  # web/utils/report_exporter.py -> project root directory

        # Get results directory configuration
        results_dir_env = os.getenv("TRADINGAGENTS_RESULTS_DIR")
        if results_dir_env:
            # If environment variable is a relative path, resolve relative to project root directory
            if not os.path.isabs(results_dir_env):
                results_dir = project_root / results_dir_env
            else:
                results_dir = Path(results_dir_env)
        else:
            # Default to using results under project root directory
            results_dir = project_root / "results"

        # Create stock-specific directory
        analysis_date = datetime.now().strftime('%Y-%m-%d')
        stock_dir = results_dir / stock_symbol / analysis_date / "reports"
        stock_dir.mkdir(parents=True, exist_ok=True)

        # Save file
        file_path = stock_dir / filename
        with open(file_path, 'wb') as f:
            f.write(content)

        logger.info(f"✅ Report saved to: {file_path}")
        logger.info(f"📁 Project root directory: {project_root}")
        logger.info(f"📁 Results directory: {results_dir}")
        logger.info(f"📁 Environment variable TRADINGAGENTS_RESULTS_DIR: {results_dir_env}")

        return str(file_path)

    except Exception as e:
        logger.error(f"❌ Failed to save report to results directory: {e}")
        import traceback
        logger.error(f"❌ Detailed error: {traceback.format_exc()}")
        return ""


def render_export_buttons(results: Dict[str, Any]):
    """Render export buttons"""

    if not results:
        return

    st.markdown("---")
    st.subheader("📤 Export Report")

    # Check if export functionality is available
    if not report_exporter.export_available:
        st.warning("⚠️ Export functionality requires additional dependency packages")
        st.code("pip install pypandoc markdown")
        return

    # Check if pandoc is available
    if not report_exporter.pandoc_available:
        st.warning("⚠️ Word and PDF export requires pandoc tool")
        st.info("💡 You can still use Markdown format export")

    # Display Docker environment status
    if report_exporter.is_docker:
        if DOCKER_ADAPTER_AVAILABLE:
            docker_status = get_docker_status_info()
            if docker_status['dependencies_ok'] and docker_status['pdf_test_ok']:
                st.success("🐳 Docker environment PDF support enabled")
            else:
                st.warning(f"🐳 Docker environment PDF support abnormal: {docker_status['dependency_message']}")
        else:
            st.warning("🐳 Docker environment detected, but adapter not available")

        with st.expander("📖 How to install pandoc"):
            st.markdown("""
            **Windows users:**
            ```bash
            # Using Chocolatey (recommended)
            choco install pandoc

            # Or download installer
            # https://github.com/jgm/pandoc/releases
            ```

            **Or use Python automatic download:**
            ```python
            import pypandoc

            pypandoc.download_pandoc()
            ```
            """)

        # In Docker environment, show all buttons even if pandoc has issues, let users try
        pass
    
    # Generate filename
    stock_symbol = results.get('stock_symbol', 'analysis')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📄 Export Markdown", help="Export as Markdown format"):
            logger.info(f"🖱️ [EXPORT] User clicked Markdown export button - Stock: {stock_symbol}")
            logger.info(f"🖱️ User clicked Markdown export button - Stock: {stock_symbol}")
            # 1. Save modular reports (CLI format)
            logger.info("📁 Starting to save modular reports (CLI format)...")
            modular_files = save_modular_reports_to_results_dir(results, stock_symbol)

            # 2. Generate summary report (for download)
            content = report_exporter.export_report(results, 'markdown')
            if content:
                filename = f"{stock_symbol}_analysis_{timestamp}.md"
                logger.info(f"✅ [EXPORT] Markdown export successful, filename: {filename}")
                logger.info(f"✅ Markdown export successful, filename: {filename}")

                # 3. Save summary report to results directory
                saved_path = save_report_to_results_dir(content, filename, stock_symbol)

                # 4. Display save results
                if modular_files and saved_path:
                    st.success(f"✅ Saved {len(modular_files)} modular reports + 1 summary report")
                    with st.expander("📁 View saved files"):
                        st.write("**Modular reports:**")
                        for module, path in modular_files.items():
                            st.write(f"- {module}: `{path}`")
                        st.write("**Summary report:**")
                        st.write(f"- Summary report: `{saved_path}`")
                elif saved_path:
                    st.success(f"✅ Summary report saved to: {saved_path}")

                st.download_button(
                    label="📥 Download Markdown",
                    data=content,
                    file_name=filename,
                    mime="text/markdown"
                )
            else:
                logger.error(f"❌ [EXPORT] Markdown export failed, content is empty")
                logger.error("❌ Markdown export failed, content is empty")
    
    with col2:
        if st.button("📝 Export Word", help="Export as Word document format"):
            logger.info(f"🖱️ [EXPORT] User clicked Word export button - Stock: {stock_symbol}")
            logger.info(f"🖱️ User clicked Word export button - Stock: {stock_symbol}")
            with st.spinner("Generating Word document, please wait..."):
                try:
                    logger.info(f"🔄 [EXPORT] Starting Word export process...")
                    logger.info("🔄 Starting Word export process...")

                    # 1. Save modular reports (CLI format)
                    logger.info("📁 Starting to save modular reports (CLI format)...")
                    modular_files = save_modular_reports_to_results_dir(results, stock_symbol)

                    # 2. Generate Word summary report
                    content = report_exporter.export_report(results, 'docx')
                    if content:
                        filename = f"{stock_symbol}_analysis_{timestamp}.docx"
                        logger.info(f"✅ [EXPORT] Word export successful, filename: {filename}, size: {len(content)} bytes")
                        logger.info(f"✅ Word export successful, filename: {filename}, size: {len(content)} bytes")

                        # 3. Save Word summary report to results directory
                        saved_path = save_report_to_results_dir(content, filename, stock_symbol)

                        # 4. Display save results
                        if modular_files and saved_path:
                            st.success(f"✅ Saved {len(modular_files)} modular reports + 1 Word summary report")
                            with st.expander("📁 View saved files"):
                                st.write("**Modular reports:**")
                                for module, path in modular_files.items():
                                    st.write(f"- {module}: `{path}`")
                                st.write("**Word summary report:**")
                                st.write(f"- Word report: `{saved_path}`")
                        elif saved_path:
                            st.success(f"✅ Word document saved to: {saved_path}")
                        else:
                            st.success("✅ Word document generated successfully!")

                        st.download_button(
                            label="📥 Download Word",
                            data=content,
                            file_name=filename,
                            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                        )
                    else:
                        logger.error(f"❌ [EXPORT] Word export failed, content is empty")
                        logger.error("❌ Word export failed, content is empty")
                        st.error("❌ Word document generation failed")
                except Exception as e:
                    logger.error(f"❌ [EXPORT] Word export exception: {str(e)}")
                    logger.error(f"❌ Word export exception: {str(e)}", exc_info=True)
                    st.error(f"❌ Word document generation failed: {str(e)}")

                    # Display detailed error information
                    with st.expander("🔍 View detailed error information"):
                        st.text(str(e))

                    # Provide solutions
                    with st.expander("💡 Solutions"):
                        st.markdown("""
                        **Word export requires pandoc tool, please check:**

                        1. **Docker environment**: Rebuild image to ensure pandoc is included
                        2. **Local environment**: Install pandoc
                        ```bash
                        # Windows
                        choco install pandoc

                        # macOS
                        brew install pandoc

                        # Linux
                        sudo apt-get install pandoc
                        ```

                        3. **Alternative**: Use Markdown format export
                        """)
    
    with col3:
        if st.button("📊 Export PDF", help="Export as PDF format (requires additional tools)"):
            logger.info(f"🖱️ User clicked PDF export button - Stock: {stock_symbol}")
            with st.spinner("Generating PDF, please wait..."):
                try:
                    logger.info("🔄 Starting PDF export process...")

                    # 1. Save modular reports (CLI format)
                    logger.info("📁 Starting to save modular reports (CLI format)...")
                    modular_files = save_modular_reports_to_results_dir(results, stock_symbol)

                    # 2. Generate PDF summary report
                    content = report_exporter.export_report(results, 'pdf')
                    if content:
                        filename = f"{stock_symbol}_analysis_{timestamp}.pdf"
                        logger.info(f"✅ PDF export successful, filename: {filename}, size: {len(content)} bytes")

                        # 3. Save PDF summary report to results directory
                        saved_path = save_report_to_results_dir(content, filename, stock_symbol)

                        # 4. Display save results
                        if modular_files and saved_path:
                            st.success(f"✅ Saved {len(modular_files)} modular reports + 1 PDF summary report")
                            with st.expander("📁 View saved files"):
                                st.write("**Modular reports:**")
                                for module, path in modular_files.items():
                                    st.write(f"- {module}: `{path}`")
                                st.write("**PDF summary report:**")
                                st.write(f"- PDF report: `{saved_path}`")
                        elif saved_path:
                            st.success(f"✅ PDF saved to: {saved_path}")
                        else:
                            st.success("✅ PDF generated successfully!")

                        st.download_button(
                            label="📥 Download PDF",
                            data=content,
                            file_name=filename,
                            mime="application/pdf"
                        )
                    else:
                        logger.error("❌ PDF export failed, content is empty")
                        st.error("❌ PDF generation failed")
                except Exception as e:
                    logger.error(f"❌ PDF export exception: {str(e)}", exc_info=True)
                    st.error(f"❌ PDF generation failed")

                    # Display detailed error information
                    with st.expander("🔍 View detailed error information"):
                        st.text(str(e))

                    # Provide solutions
                    with st.expander("💡 Solutions"):
                        st.markdown("""
                        **PDF export requires additional tools, please choose one of the following options:**

                        **Option 1: Install wkhtmltopdf (Recommended)**
                        ```bash
                        # Windows
                        choco install wkhtmltopdf

                        # macOS
                        brew install wkhtmltopdf

                        # Linux
                        sudo apt-get install wkhtmltopdf
                        ```

                        **Option 2: Install LaTeX**
                        ```bash
                        # Windows
                        choco install miktex

                        # macOS
                        brew install mactex

                        # Linux
                        sudo apt-get install texlive-full
                        ```

                        **Option 3: Use alternative formats**
                        - 📄 Markdown format - Lightweight, good compatibility
                        - 📝 Word format - Suitable for further editing
                        """)

                    # Suggest using other formats
                    st.info("💡 Suggestion: You can first export in Markdown or Word format, then use other tools to convert to PDF")


def save_analysis_report(stock_symbol: str, analysis_results: Dict[str, Any], 
                        report_content: str = None) -> bool:
    """
    Save analysis report to MongoDB
    
    Args:
        stock_symbol: Stock symbol
        analysis_results: Analysis results dictionary
        report_content: Report content (optional, will be auto-generated if not provided)
    
    Returns:
        bool: Whether the save was successful
    """
    try:
        if not MONGODB_REPORT_AVAILABLE or mongodb_report_manager is None:
            logger.warning("MongoDB report manager not available, cannot save report")
            return False
        
        # If no report content is provided, generate Markdown report
        if report_content is None:
            report_content = report_exporter.generate_markdown_report(analysis_results)
        
        # Call MongoDB report manager to save report
        # Wrap report content in dictionary format
        reports_dict = {
            "markdown": report_content,
            "generated_at": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        success = mongodb_report_manager.save_analysis_report(
            stock_symbol=stock_symbol,
            analysis_results=analysis_results,
            reports=reports_dict
        )
        
        if success:
            logger.info(f"✅ Analysis report successfully saved to MongoDB - Stock: {stock_symbol}")
        else:
            logger.error(f"❌ Failed to save analysis report to MongoDB - Stock: {stock_symbol}")
        
        return success
        
    except Exception as e:
        logger.error(f"❌ Exception occurred while saving analysis report to MongoDB - Stock: {stock_symbol}, Error: {str(e)}")
        return False
    
 