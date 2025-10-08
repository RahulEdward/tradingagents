# TradingAgents Enhanced Version

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Version](https://img.shields.io/badge/Version-cn--0.1.15-green.svg)](./VERSION)
[![Documentation](https://img.shields.io/badge/docs-English%20Documentation-green.svg)](./docs/)
[![Original](https://img.shields.io/badge/Based%20on-TauricResearch/TradingAgents-orange.svg)](https://github.com/TauricResearch/TradingAgents)

> 🚀 **Latest Version cn-0.1.15**: Major upgrade in developer experience and LLM ecosystem! Added Qianfan model support, complete development toolchain, academic research materials, and enterprise-grade workflow standards!
>
> 🎯 **Core Features**: Native OpenAI Support | Comprehensive Google AI Integration | Custom Endpoint Configuration | Intelligent Model Selection | Multi-LLM Provider Support | Model Selection Persistence | Docker Containerized Deployment | Professional Report Export | Complete A-Share Support | Chinese Localization

A **Chinese financial trading decision framework** based on multi-agent large language models. Optimized for Chinese users, providing comprehensive analysis capabilities for A-shares/Hong Kong stocks/US stocks.

## 🙏 Tribute to the Original Project

Thanks to the [py-genie](https://github.com/py-genie) team for creating the revolutionary multi-agent trading framework [TradingAgents](https://github.com/py-genie/TradingAgents)!

**🎯 Our Mission**: To provide Chinese users with a complete Chinese experience, support A-share/Hong Kong stock markets, integrate domestic large models, and promote the popularization of AI financial technology in the Chinese community.

## 🆕 v0.1.15 Major Updates

### 🤖 LLM Ecosystem Major Upgrade

- **Qianfan Model Support**: Added complete integration of Baidu Qianfan (ERNIE) large models
- **LLM Adapter Refactoring**: Unified OpenAI-compatible adapter architecture
- **Multi-vendor Support**: Support for more domestic large model providers
- **Integration Guide**: Complete LLM integration development documentation and testing tools

### 📚 Academic Research Support

- **TradingAgents Paper**: Complete Chinese translation and in-depth interpretation
- **Technical Blog**: Detailed technical analysis and implementation principle interpretation
- **Academic Materials**: PDF papers and related research materials
- **Citation Support**: Standard academic citation format and references

### 🛠️ Developer Experience Upgrade

- **Development Workflow**: Standardized development process and branch management specifications
- **Installation Verification**: Complete installation testing and verification scripts
- **Documentation Refactoring**: Structured documentation system and quick start guide
- **PR Templates**: Standardized Pull Request templates and code review process

### 🔧 Enterprise-grade Toolchain

- **Branch Protection**: GitHub branch protection strategies and security rules
- **Emergency Procedures**: Complete emergency handling and disaster recovery procedures
- **Testing Framework**: Enhanced test coverage and verification tools
- **Deployment Guide**: Enterprise-grade deployment and configuration management

## 📋 v0.1.14 Feature Review

### 👥 User Permission Management System

- **Complete User Management**: Added user registration, login, and permission control functions
- **Role Permissions**: Support for multi-level user roles and permission management
- **Session Management**: Secure user sessions and state management
- **User Activity Logs**: Complete user operation records and audit functions

### 🔐 Web User Authentication System

- **Login Components**: Modern user login interface
- **Authentication Manager**: Unified user authentication and authorization management
- **Security Enhancement**: Security mechanisms such as password encryption and session security
- **User Dashboard**: Personalized user activity dashboard

### 🗄️ Data Management Optimization

- **Enhanced MongoDB Integration**: Improved MongoDB connection and data management
- **Data Directory Reorganization**: Optimized data storage structure and management
- **Data Migration Scripts**: Complete data migration and backup tools
- **Cache Optimization**: Improved data loading and analysis result cache performance

### 🧪 Enhanced Test Coverage

- **Functional Test Scripts**: Added 6 specialized functional test scripts
- **Tool Processor Testing**: Google tool processor fix verification
- **Auto-hide Guide Testing**: UI interaction function testing
- **Online Tool Configuration Testing**: Tool configuration and selection logic testing
- **Real Scenario Testing**: End-to-end testing of actual usage scenarios
- **US Stock Independence Testing**: US stock analysis function independence verification

---

## 🆕 v0.1.13 Major Updates

### 🤖 Native OpenAI Endpoint Support

- **Custom OpenAI Endpoints**: Support for configuring any OpenAI-compatible API endpoints
- **Flexible Model Selection**: Can use any OpenAI format models, not limited to official models
- **Intelligent Adapter**: Added native OpenAI adapter for better compatibility and performance
- **Configuration Management**: Unified endpoint and model configuration management system

### 🧠 Comprehensive Google AI Ecosystem Integration

- **Three Google AI Package Support**: langchain-google-genai, google-generativeai, google-genai
- **9 Verified Models**: gemini-2.5-pro, gemini-2.5-flash, gemini-2.0-flash and other latest models
- **Google Tool Processor**: Dedicated Google AI tool calling processor
- **Intelligent Degradation Mechanism**: Automatic degradation to basic functions when advanced features fail

### 🔧 LLM Adapter Architecture Optimization

- **GoogleOpenAIAdapter**: Added Google AI's OpenAI-compatible adapter
- **Unified Interface**: All LLM providers use unified calling interface
- **Enhanced Error Handling**: Improved exception handling and automatic retry mechanism
- **Performance Monitoring**: Added LLM call performance monitoring and statistics

### 🎨 Intelligent Web Interface Optimization

- **Intelligent Model Selection**: Automatically select the best model based on availability
- **KeyError Fix**: Completely resolved KeyError issues in model selection
- **UI Response Optimization**: Improved response speed and user experience for model switching
- **Error Prompts**: More friendly error prompts and solution suggestions

## 🆕 v0.1.12 Major Updates

### 🧠 Intelligent News Analysis Module

- **Intelligent News Filter**: AI-based news relevance scoring and quality assessment
- **Multi-level Filtering Mechanism**: Three-tier processing with basic, enhanced, and integrated filtering
- **News Quality Assessment**: Automatic identification and filtering of low-quality, duplicate, and irrelevant news
- **Unified News Tool**: Integrates multiple news sources, providing unified news retrieval interface

### 🔧 Technical Fixes and Optimizations

- **DashScope Adapter Fix**: Resolved tool calling compatibility issues
- **DeepSeek Infinite Loop Fix**: Fixed infinite loop issues in news analysts
- **Enhanced LLM Tool Calling**: Improved reliability and stability of tool calls
- **News Retriever Optimization**: Enhanced news data acquisition and processing capabilities

### 📚 Comprehensive Testing and Documentation

- **Comprehensive Test Coverage**: Added 15+ test files covering all new features
- **Detailed Technical Documentation**: Added 8 technical analysis reports and fix documentation
- **User Guide Enhancement**: Added news filtering usage guide and best practices
- **Demo Scripts**: Provided complete news filtering functionality demonstrations

### 🗂️ Project Structure Optimization

- **Documentation Classification**: Organized documents by function into docs subdirectories
- **Example Code Organization**: Unified demo scripts to examples directory
- **Clean Root Directory**: Maintained clean root directory, improved project professionalism

## 🎯 Core Features

### 🤖 Multi-Agent Collaborative Architecture

- **Professional Division**: Four major analysts for fundamentals, technicals, news, and social media
- **Structured Debate**: Bullish/bearish researchers conduct in-depth analysis
- **Intelligent Decision Making**: Traders make final investment recommendations based on all inputs
- **Risk Management**: Multi-level risk assessment and management mechanisms

## 🖥️ Web Interface Showcase

### 📸 Interface Screenshots

> 🎨 **Modern Web Interface**: Responsive web application built on Streamlit, providing intuitive stock analysis experience

#### 🏠 Main Interface - Analysis Configuration

![1755003162925](images/README/1755003162925.png)

![1755002619976](images/README/1755002619976.png)

*Intelligent configuration panel supporting multi-market stock analysis with 5-level research depth selection*

#### 📊 Real-time Analysis Progress

![1755002731483](images/README/1755002731483.png)

*Real-time progress tracking, visualized analysis process, intelligent time estimation*

#### 📈 Analysis Results Display

![1755002901204](images/README/1755002901204.png)

![1755002924844](images/README/1755002924844.png)

![1755002939905](images/README/1755002939905.png)

![1755002968608](images/README/1755002968608.png)

![1755002985903](images/README/1755002985903.png)

![1755003004403](images/README/1755003004403.png)

![1755003019759](images/README/1755003019759.png)

![1755003033939](images/README/1755003033939.png)

![1755003048242](images/README/1755003048242.png)

![1755003064598](images/README/1755003064598.png)

![1755003090603](images/README/1755003090603.png)

*Professional investment reports, multi-dimensional analysis results, one-click export functionality*

### 🎯 Core Feature Highlights

#### 📋 **Intelligent Analysis Configuration**

- **🌍 Multi-Market Support**: One-stop analysis for US stocks, A-shares, and Hong Kong stocks
- **🎯 5-Level Research Depth**: From 2-minute quick analysis to 25-minute comprehensive research
- **🤖 Agent Selection**: Market technical, fundamental, news, and social media analysts
- **📅 Flexible Time Settings**: Support for analysis at any historical time point

#### 🚀 **Real-time Progress Tracking**

- **📊 Visual Progress**: Real-time display of analysis progress and remaining time
- **🔄 Intelligent Step Recognition**: Automatic identification of current analysis stage
- **⏱️ Accurate Time Estimation**: Intelligent time calculation based on historical data
- **💾 State Persistence**: Analysis progress not lost on page refresh

#### 📈 **Professional Results Display**

- **🎯 Investment Decisions**: Clear buy/hold/sell recommendations
- **📊 Multi-dimensional Analysis**: Comprehensive evaluation of technical, fundamental, and news aspects
- **🔢 Quantitative Indicators**: Confidence levels, risk scores, target prices
- **📄 Professional Reports**: Support for Markdown/Word/PDF format export

#### 🤖 **Multi-LLM Model Management**

- **🌐 4 Major Providers**: DashScope, DeepSeek, Google AI, OpenRouter
- **🎯 60+ Model Selection**: Full coverage from economical to flagship models
- **💾 Configuration Persistence**: URL parameter storage, settings maintained on refresh
- **⚡ Quick Switching**: 5 popular model one-click selection buttons

### 🎮 Web Interface Operation Guide

#### 🚀 **Quick Start Process**

1. **Launch Application**: `python start_web.py` or `docker-compose up -d`
2. **Access Interface**: Open browser to `http://localhost:8501`
3. **Configure Model**: Select LLM provider and model in sidebar
4. **Input Stock**: Enter stock code (e.g., AAPL, 000001, 0700.HK)
5. **Select Depth**: Choose 1-5 level research depth based on needs
6. **Start Analysis**: Click "🚀 Start Analysis" button
7. **View Results**: Track progress in real-time, view analysis report
8. **Export Report**: One-click export professional format reports

#### 📊 **Supported Stock Code Formats**

- **🇺🇸 US Stocks**: `AAPL`, `TSLA`, `MSFT`, `NVDA`, `GOOGL`
- **🇨🇳 A-Shares**: `000001`, `600519`, `300750`, `002415`
- **🇭🇰 Hong Kong Stocks**: `0700.HK`, `9988.HK`, `3690.HK`, `1810.HK`

#### 🎯 **Research Depth Explanation**

- **Level 1 (2-4 minutes)**: Quick overview, basic technical indicators
- **Level 2 (4-6 minutes)**: Standard analysis, technical + fundamental
- **Level 3 (6-10 minutes)**: Deep analysis, including news sentiment ⭐ **Recommended**
- **Level 4 (10-15 minutes)**: Comprehensive analysis, multi-round agent debate
- **Level 5 (15-25 minutes)**: Deepest analysis, complete research report

#### 💡 **Usage Tips**

- **🔄 Real-time Refresh**: Can refresh page anytime during analysis, progress not lost
- **📱 Mobile Adaptation**: Supports mobile and tablet device access
- **🎨 Dark Mode**: Automatically adapts to system theme settings
- **⌨️ Shortcuts**: Supports Enter key for quick analysis submission
- **📋 History**: Automatically saves recent analysis configurations

> 📖 **Detailed Guide**: Complete web interface usage instructions available at [🖥️ Web Interface Detailed Usage Guide](docs/usage/web-interface-detailed-guide.md)