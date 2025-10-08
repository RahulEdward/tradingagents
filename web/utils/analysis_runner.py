"""
Stock analysis execution tool
"""

import sys
import os
import uuid
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# Import logging modules
from tradingagents.utils.logging_manager import get_logger, get_logger_manager
logger = get_logger('web')

# Add project root directory to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Ensure environment variables are loaded correctly
load_dotenv(project_root / ".env", override=True)

# Import unified logging system
from tradingagents.utils.logging_init import setup_web_logging
logger = setup_web_logging()

# Add configuration manager
try:
    from tradingagents.config.config_manager import token_tracker
    TOKEN_TRACKING_ENABLED = True
    logger.info("✅ Token tracking feature enabled")
except ImportError:
    TOKEN_TRACKING_ENABLED = False
    logger.warning("⚠️ Token tracking feature not enabled")

def translate_analyst_labels(text):
    """Convert English analyst labels to Chinese"""
    if not text:
        return text

    # Analyst label translation mapping (Chinese to English)
    translations = {
        '看涨分析师:': 'Bull Analyst:',
        '看跌分析师:': 'Bear Analyst:',
        '激进风险分析师:': 'Risky Analyst:',
        '保守风险分析师:': 'Safe Analyst:',
        '中性风险分析师:': 'Neutral Analyst:',
        '研究经理:': 'Research Manager:',
        '投资组合经理:': 'Portfolio Manager:',
        '风险管理委员会:': 'Risk Judge:',
        '交易员:': 'Trader:'
    }

    # Replace all Chinese labels with English
    for chinese, english in translations.items():
        text = text.replace(chinese, english)

    return text

def extract_risk_assessment(state):
    """Extract risk assessment data from analysis state"""
    try:
        risk_debate_state = state.get('risk_debate_state', {})

        if not risk_debate_state:
            return None

        # Extract views from various risk analysts and convert to Chinese
        risky_analysis = translate_analyst_labels(risk_debate_state.get('risky_history', ''))
        safe_analysis = translate_analyst_labels(risk_debate_state.get('safe_history', ''))
        neutral_analysis = translate_analyst_labels(risk_debate_state.get('neutral_history', ''))
        judge_decision = translate_analyst_labels(risk_debate_state.get('judge_decision', ''))

        # Format risk assessment report
        risk_assessment = f"""
## ⚠️ Risk Assessment Report

### 🔴 Risky Analyst Perspective
{risky_analysis if risky_analysis else 'No risky analysis available'}

### 🟡 Neutral Analyst Perspective
{neutral_analysis if neutral_analysis else 'No neutral analysis available'}

### 🟢 Safe Analyst Perspective
{safe_analysis if safe_analysis else 'No safe analysis available'}

### 🏛️ Risk Management Committee Final Decision
{judge_decision if judge_decision else 'No risk management decision available'}

---
*Risk assessment is based on multi-perspective analysis. Please make investment decisions based on your personal risk tolerance.*
        """.strip()

        return risk_assessment

    except Exception as e:
        logger.info(f"Error extracting risk assessment data: {e}")
        return None

def run_stock_analysis(stock_symbol, analysis_date, analysts, research_depth, llm_provider, llm_model, market_type="US Stock", progress_callback=None):
    """Execute stock analysis

    Args:
        stock_symbol: Stock symbol
        analysis_date: Analysis date
        analysts: List of analysts
        research_depth: Research depth
        llm_provider: LLM provider (dashscope/deepseek/google)
        llm_model: LLM model name
        progress_callback: Progress callback function for updating UI status
    """

    def update_progress(message, step=None, total_steps=None):
        """Update progress"""
        if progress_callback:
            progress_callback(message, step, total_steps)
        logger.info(f"[Progress] {message}")

    # Generate session ID for token tracking and log correlation
    session_id = f"analysis_{uuid.uuid4().hex[:8]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # 1. Data pre-fetching and validation phase
    update_progress("🔍 Validating stock code and pre-fetching data...", 1, 10)

    try:
        from tradingagents.utils.stock_validator import prepare_stock_data

        # Pre-fetch stock data (default 30 days historical data)
        preparation_result = prepare_stock_data(
            stock_code=stock_symbol,
            market_type=market_type,
            period_days=30,  # Can be adjusted based on research_depth
            analysis_date=analysis_date
        )

        if not preparation_result.is_valid:
            error_msg = f"❌ Stock data validation failed: {preparation_result.error_message}"
            update_progress(error_msg)
            logger.error(f"[{session_id}] {error_msg}")

            return {
                'success': False,
                'error': preparation_result.error_message,
                'suggestion': preparation_result.suggestion,
                'stock_symbol': stock_symbol,
                'analysis_date': analysis_date,
                'session_id': session_id
            }

        # Data pre-fetching successful
        success_msg = f"✅ Data preparation completed: {preparation_result.stock_name} ({preparation_result.market_type})"
        update_progress(success_msg)  # Use intelligent detection, no longer hardcode steps
        logger.info(f"[{session_id}] {success_msg}")
        logger.info(f"[{session_id}] Cache status: {preparation_result.cache_status}")

    except Exception as e:
        error_msg = f"❌ Error occurred during data pre-fetching: {str(e)}"
        update_progress(error_msg)
        logger.error(f"[{session_id}] {error_msg}")

        return {
            'success': False,
            'error': error_msg,
            'suggestion': "Please check network connection or try again later",
            'stock_symbol': stock_symbol,
            'analysis_date': analysis_date,
            'session_id': session_id
        }

    # Record detailed logs for analysis start
    logger_manager = get_logger_manager()
    import time
    analysis_start_time = time.time()

    logger_manager.log_analysis_start(
        logger, stock_symbol, "comprehensive_analysis", session_id
    )

    logger.info(f"🚀 [Analysis Start] Stock analysis initiated",
               extra={
                   'stock_symbol': stock_symbol,
                   'analysis_date': analysis_date,
                   'analysts': analysts,
                   'research_depth': research_depth,
                   'llm_provider': llm_provider,
                   'llm_model': llm_model,
                   'market_type': market_type,
                   'session_id': session_id,
                   'event_type': 'web_analysis_start'
               })

    update_progress("🚀 Starting stock analysis...")

    # Estimate token usage (for cost estimation)
    if TOKEN_TRACKING_ENABLED:
        estimated_input = 2000 * len(analysts)  # Estimate 2000 input tokens per analyst
        estimated_output = 1000 * len(analysts)  # Estimate 1000 output tokens per analyst
        estimated_cost = token_tracker.estimate_cost(llm_provider, llm_model, estimated_input, estimated_output)

        update_progress(f"💰 Estimated analysis cost: ¥{estimated_cost:.4f}")

    # Validate environment variables
    update_progress("Checking environment variable configuration...")
    dashscope_key = os.getenv("DASHSCOPE_API_KEY")
    finnhub_key = os.getenv("FINNHUB_API_KEY")

    logger.info(f"Environment variable check:")
    logger.info(f"  DASHSCOPE_API_KEY: {'Set' if dashscope_key else 'Not set'}")
    logger.info(f"  FINNHUB_API_KEY: {'Set' if finnhub_key else 'Not set'}")

    if not dashscope_key:
        raise ValueError("DASHSCOPE_API_KEY environment variable not set")
    if not finnhub_key:
        raise ValueError("FINNHUB_API_KEY environment variable not set")

    update_progress("Environment variable validation passed")

    try:
        # Import necessary modules
        from tradingagents.graph.trading_graph import TradingAgentsGraph
        from tradingagents.default_config import DEFAULT_CONFIG

        # Create configuration
        update_progress("Configuring analysis parameters...")
        config = DEFAULT_CONFIG.copy()
        config["llm_provider"] = llm_provider
        config["deep_think_llm"] = llm_model
        config["quick_think_llm"] = llm_model
        # Adjust configuration based on research depth
        if research_depth == 1:  # Level 1 - Quick analysis
            config["max_debate_rounds"] = 1
            config["max_risk_discuss_rounds"] = 1
            # Keep memory function enabled, as memory operations have minimal overhead but significantly improve analysis quality
            config["memory_enabled"] = True

            # Use online tools uniformly to avoid various issues with offline tools
            config["online_tools"] = True  # All markets use unified tools
            logger.info(f"🔧 [Quick Analysis] {market_type} using unified tools to ensure correct data sources and stability")
            if llm_provider == "dashscope":
                config["quick_think_llm"] = "qwen-turbo"  # Use fastest model
                config["deep_think_llm"] = "qwen-plus"
            elif llm_provider == "deepseek":
                config["quick_think_llm"] = "deepseek-chat"  # DeepSeek只有一个模型
                config["deep_think_llm"] = "deepseek-chat"
        elif research_depth == 2:  # 2级 - 基础分析
            config["max_debate_rounds"] = 1
            config["max_risk_discuss_rounds"] = 1
            config["memory_enabled"] = True
            config["online_tools"] = True
            if llm_provider == "dashscope":
                config["quick_think_llm"] = "qwen-plus"
                config["deep_think_llm"] = "qwen-plus"
            elif llm_provider == "deepseek":
                config["quick_think_llm"] = "deepseek-chat"
                config["deep_think_llm"] = "deepseek-chat"
            elif llm_provider == "openai":
                config["quick_think_llm"] = llm_model
                config["deep_think_llm"] = llm_model
            elif llm_provider == "openai":
                config["quick_think_llm"] = llm_model
                config["deep_think_llm"] = llm_model
            elif llm_provider == "openai":
                config["quick_think_llm"] = llm_model
                config["deep_think_llm"] = llm_model
            elif llm_provider == "openai":
                config["quick_think_llm"] = llm_model
                config["deep_think_llm"] = llm_model
            elif llm_provider == "openai":
                config["quick_think_llm"] = llm_model
                config["deep_think_llm"] = llm_model
            elif llm_provider == "openai":
                config["quick_think_llm"] = llm_model
                config["deep_think_llm"] = llm_model
        elif research_depth == 3:  # 3级 - 标准分析 (默认)
            config["max_debate_rounds"] = 1
            config["max_risk_discuss_rounds"] = 2
            config["memory_enabled"] = True
            config["online_tools"] = True
            if llm_provider == "dashscope":
                config["quick_think_llm"] = "qwen-plus"
                config["deep_think_llm"] = "qwen-max"
            elif llm_provider == "deepseek":
                config["quick_think_llm"] = "deepseek-chat"
                config["deep_think_llm"] = "deepseek-chat"
        elif research_depth == 4:  # 4级 - 深度分析
            config["max_debate_rounds"] = 2
            config["max_risk_discuss_rounds"] = 2
            config["memory_enabled"] = True
            config["online_tools"] = True
            if llm_provider == "dashscope":
                config["quick_think_llm"] = "qwen-plus"
                config["deep_think_llm"] = "qwen-max"
            elif llm_provider == "deepseek":
                config["quick_think_llm"] = "deepseek-chat"
                config["deep_think_llm"] = "deepseek-chat"
        else:  # 5级 - 全面分析
            config["max_debate_rounds"] = 3
            config["max_risk_discuss_rounds"] = 3
            config["memory_enabled"] = True
            config["online_tools"] = True
            if llm_provider == "dashscope":
                config["quick_think_llm"] = "qwen-max"
                config["deep_think_llm"] = "qwen-max"
            elif llm_provider == "deepseek":
                config["quick_think_llm"] = "deepseek-chat"
                config["deep_think_llm"] = "deepseek-chat"

        # Set different configurations based on LLM provider
        if llm_provider == "dashscope":
            config["backend_url"] = "https://dashscope.aliyuncs.com/api/v1"
        elif llm_provider == "deepseek":
            config["backend_url"] = "https://api.deepseek.com"
        elif llm_provider == "qianfan":
            # Qianfan (ERNIE) configuration
            config["backend_url"] = "https://aip.baidubce.com"
            # Set Qianfan models based on research depth
            if research_depth <= 2:  # Quick and basic analysis
                config["quick_think_llm"] = "ernie-3.5-8k"
                config["deep_think_llm"] = "ernie-3.5-8k"
            elif research_depth <= 4:  # Standard and deep analysis
                config["quick_think_llm"] = "ernie-3.5-8k"
                config["deep_think_llm"] = "ernie-4.0-turbo-8k"
            else:  # Comprehensive analysis
                config["quick_think_llm"] = "ernie-4.0-turbo-8k"
                config["deep_think_llm"] = "ernie-4.0-turbo-8k"
            
            logger.info(f"🤖 [Qianfan] Quick model: {config['quick_think_llm']}")
            logger.info(f"🤖 [Qianfan] Deep model: {config['deep_think_llm']}")
        elif llm_provider == "google":
            # Google AI doesn't need backend_url, use default OpenAI format
            config["backend_url"] = "https://api.openai.com/v1"
            
            # Optimize Google model selection based on research depth
            if research_depth == 1:  # Quick analysis - use fastest model
                config["quick_think_llm"] = "gemini-2.5-flash-lite-preview-06-17"  # 1.45s
                config["deep_think_llm"] = "gemini-2.0-flash"  # 1.87s
            elif research_depth == 2:  # Basic analysis - use fast model
                config["quick_think_llm"] = "gemini-2.0-flash"  # 1.87s
                config["deep_think_llm"] = "gemini-1.5-pro"  # 2.25s
            elif research_depth == 3:  # Standard analysis - balanced performance
                config["quick_think_llm"] = "gemini-1.5-pro"  # 2.25s
                config["deep_think_llm"] = "gemini-2.5-flash"  # 2.73s
            elif research_depth == 4:  # Deep analysis - use powerful model
                config["quick_think_llm"] = "gemini-2.5-flash"  # 2.73s
                config["deep_think_llm"] = "gemini-2.5-pro"  # 16.68s
            else:  # Comprehensive analysis - use strongest model
                config["quick_think_llm"] = "gemini-2.5-pro"  # 16.68s
                config["deep_think_llm"] = "gemini-2.5-pro"  # 16.68s
            
            logger.info(f"🤖 [Google AI] 快速模型: {config['quick_think_llm']}")
            logger.info(f"🤖 [Google AI] 深度模型: {config['deep_think_llm']}")
        elif llm_provider == "openai":
            # OpenAI官方API
            config["backend_url"] = "https://api.openai.com/v1"
            logger.info(f"🤖 [OpenAI] 使用模型: {llm_model}")
            logger.info(f"🤖 [OpenAI] API端点: https://api.openai.com/v1")
        elif llm_provider == "openrouter":
            # OpenRouter使用OpenAI兼容API
            config["backend_url"] = "https://openrouter.ai/api/v1"
            logger.info(f"🌐 [OpenRouter] 使用模型: {llm_model}")
            logger.info(f"🌐 [OpenRouter] API端点: https://openrouter.ai/api/v1")
        elif llm_provider == "siliconflow":
            config["backend_url"] = "https://api.siliconflow.cn/v1"
            logger.info(f"🌐 [SiliconFlow] 使用模型: {llm_model}")
            logger.info(f"🌐 [SiliconFlow] API端点: https://api.siliconflow.cn/v1")
        elif llm_provider == "custom_openai":
            # 自定义OpenAI端点
            custom_base_url = st.session_state.get("custom_openai_base_url", "https://api.openai.com/v1")
            config["backend_url"] = custom_base_url
            config["custom_openai_base_url"] = custom_base_url
            logger.info(f"🔧 [自定义OpenAI] 使用模型: {llm_model}")
            logger.info(f"🔧 [自定义OpenAI] API端点: {custom_base_url}")

        # 修复路径问题 - 优先使用环境变量配置
        # 数据目录：优先使用环境变量，否则使用默认路径
        if not config.get("data_dir") or config["data_dir"] == "./data":
            env_data_dir = os.getenv("TRADINGAGENTS_DATA_DIR")
            if env_data_dir:
                # 如果环境变量是相对路径，相对于项目根目录解析
                if not os.path.isabs(env_data_dir):
                    config["data_dir"] = str(project_root / env_data_dir)
                else:
                    config["data_dir"] = env_data_dir
            else:
                config["data_dir"] = str(project_root / "data")

        # 结果目录：优先使用环境变量，否则使用默认路径
        if not config.get("results_dir") or config["results_dir"] == "./results":
            env_results_dir = os.getenv("TRADINGAGENTS_RESULTS_DIR")
            if env_results_dir:
                # 如果环境变量是相对路径，相对于项目根目录解析
                if not os.path.isabs(env_results_dir):
                    config["results_dir"] = str(project_root / env_results_dir)
                else:
                    config["results_dir"] = env_results_dir
            else:
                config["results_dir"] = str(project_root / "results")

        # 缓存目录：优先使用环境变量，否则使用默认路径
        if not config.get("data_cache_dir"):
            env_cache_dir = os.getenv("TRADINGAGENTS_CACHE_DIR")
            if env_cache_dir:
                # 如果环境变量是相对路径，相对于项目根目录解析
                if not os.path.isabs(env_cache_dir):
                    config["data_cache_dir"] = str(project_root / env_cache_dir)
                else:
                    config["data_cache_dir"] = env_cache_dir
            else:
                config["data_cache_dir"] = str(project_root / "tradingagents" / "dataflows" / "data_cache")

        # Ensure directories exist
        update_progress("📁 Creating necessary directories...")
        os.makedirs(config["data_dir"], exist_ok=True)
        os.makedirs(config["results_dir"], exist_ok=True)
        os.makedirs(config["data_cache_dir"], exist_ok=True)

        logger.info(f"📁 Directory configuration:")
        logger.info(f"  - Data directory: {config['data_dir']}")
        logger.info(f"  - Results directory: {config['results_dir']}")
        logger.info(f"  - Cache directory: {config['data_cache_dir']}")
        logger.info(f"  - Environment variable TRADINGAGENTS_RESULTS_DIR: {os.getenv('TRADINGAGENTS_RESULTS_DIR', 'Not set')}")

        logger.info(f"Using configuration: {config}")
        logger.info(f"Analyst list: {analysts}")
        logger.info(f"Stock symbol: {stock_symbol}")
        logger.info(f"Analysis date: {analysis_date}")

        # Adjust stock symbol format based on market type
        logger.debug(f"🔍 [RUNNER DEBUG] ===== Stock Symbol Formatting =====")
        logger.debug(f"🔍 [RUNNER DEBUG] Original stock symbol: '{stock_symbol}'")
        logger.debug(f"🔍 [RUNNER DEBUG] Market type: '{market_type}'")

        if market_type == "A股":
            # A-share codes don't need special handling, keep as is
            formatted_symbol = stock_symbol
            logger.debug(f"🔍 [RUNNER DEBUG] A-share code kept as is: '{formatted_symbol}'")
            update_progress(f"🇨🇳 Preparing to analyze A-share: {formatted_symbol}")
        elif market_type == "港股":
            # H-share codes converted to uppercase, ensure .HK suffix
            formatted_symbol = stock_symbol.upper()
            if not formatted_symbol.endswith('.HK'):
                # If it's pure digits, add .HK suffix
                if formatted_symbol.isdigit():
                    formatted_symbol = f"{formatted_symbol.zfill(4)}.HK"
            update_progress(f"🇭🇰 Preparing to analyze H-share: {formatted_symbol}")
        else:
            # US stock codes converted to uppercase
            formatted_symbol = stock_symbol.upper()
            logger.debug(f"🔍 [RUNNER DEBUG] US stock code converted to uppercase: '{stock_symbol}' -> '{formatted_symbol}'")
            update_progress(f"🇺🇸 Preparing to analyze US stock: {formatted_symbol}")

        logger.debug(f"🔍 [RUNNER DEBUG] Final stock symbol passed to analysis engine: '{formatted_symbol}'")

        # Initialize trading graph
        update_progress("🔧 Initializing analysis engine...")
        graph = TradingAgentsGraph(analysts, config=config, debug=False)

        # Execute analysis
        update_progress(f"📊 Starting analysis of {formatted_symbol} stock, this may take a few minutes...")
        logger.debug(f"🔍 [RUNNER DEBUG] ===== Calling graph.propagate =====")
        logger.debug(f"🔍 [RUNNER DEBUG] Parameters passed to graph.propagate:")
        logger.debug(f"🔍 [RUNNER DEBUG]   symbol: '{formatted_symbol}'")
        logger.debug(f"🔍 [RUNNER DEBUG]   date: '{analysis_date}'")

        state, decision = graph.propagate(formatted_symbol, analysis_date)

        # Debug information
        logger.debug(f"🔍 [DEBUG] Analysis completed, decision type: {type(decision)}")
        logger.debug(f"🔍 [DEBUG] decision content: {decision}")

        # Format results
        update_progress("📋 Analysis completed, organizing results...")

        # Extract risk assessment data
        risk_assessment = extract_risk_assessment(state)

        # Add risk assessment to state
        if risk_assessment:
            state['risk_assessment'] = risk_assessment

        # Record token usage (actual usage, using estimates here)
        if TOKEN_TRACKING_ENABLED:
            # In actual application, these values should be obtained from LLM responses
            # Here using estimates based on number of analysts and research depth
            actual_input_tokens = len(analysts) * (1500 if research_depth == "快速" else 2500 if research_depth == "标准" else 4000)
            actual_output_tokens = len(analysts) * (800 if research_depth == "快速" else 1200 if research_depth == "标准" else 2000)

            usage_record = token_tracker.track_usage(
                provider=llm_provider,
                model_name=llm_model,
                input_tokens=actual_input_tokens,
                output_tokens=actual_output_tokens,
                session_id=session_id,
                analysis_type=f"{market_type}_analysis"
            )

            if usage_record:
                update_progress(f"💰 Recorded usage cost: ¥{usage_record.cost:.4f}")

        results = {
            'stock_symbol': stock_symbol,
            'analysis_date': analysis_date,
            'analysts': analysts,
            'research_depth': research_depth,
            'llm_provider': llm_provider,
            'llm_model': llm_model,
            'state': state,
            'decision': decision,
            'success': True,
            'error': None,
            'session_id': session_id if TOKEN_TRACKING_ENABLED else None
        }

        # Record detailed log of analysis completion
        analysis_duration = time.time() - analysis_start_time

        # Calculate total cost (if token tracking enabled)
        total_cost = 0.0
        if TOKEN_TRACKING_ENABLED:
            try:
                total_cost = token_tracker.get_session_cost(session_id)
            except:
                pass

        logger_manager.log_analysis_complete(
            logger, stock_symbol, "comprehensive_analysis", session_id,
            analysis_duration, total_cost
        )

        logger.info(f"✅ [Analysis Complete] Stock analysis successfully completed",
                   extra={
                       'stock_symbol': stock_symbol,
                       'session_id': session_id,
                       'duration': analysis_duration,
                       'total_cost': total_cost,
                       'analysts_used': analysts,
                       'success': True,
                       'event_type': 'web_analysis_complete'
                   })

        # Save analysis report to local and MongoDB
        try:
            update_progress("💾 Saving analysis report...")
            from .report_exporter import save_analysis_report, save_modular_reports_to_results_dir
            
            # 1. Save modular reports to local directory
            logger.info(f"📁 [Local Save] Starting to save modular reports to local directory")
            local_files = save_modular_reports_to_results_dir(results, stock_symbol)
            if local_files:
                logger.info(f"✅ [Local Save] Saved {len(local_files)} local report files")
                for module, path in local_files.items():
                    logger.info(f"  - {module}: {path}")
            else:
                logger.warning(f"⚠️ [Local Save] Local report file save failed")
            
            # 2. Save analysis report to MongoDB
            logger.info(f"🗄️ [MongoDB Save] Starting to save analysis report to MongoDB")
            save_success = save_analysis_report(
                stock_symbol=stock_symbol,
                analysis_results=results
            )
            
            if save_success:
                logger.info(f"✅ [MongoDB Save] Analysis report successfully saved to MongoDB")
                update_progress("✅ Analysis report saved to database and local files")
            else:
                logger.warning(f"⚠️ [MongoDB Save] MongoDB report save failed")
                if local_files:
                    update_progress("✅ Local report saved, but database save failed")
                else:
                    update_progress("⚠️ Report save failed, but analysis completed")
                
        except Exception as save_error:
            logger.error(f"❌ [Report Save] Error occurred while saving analysis report: {str(save_error)}")
            update_progress("⚠️ Report save error, but analysis completed")

        update_progress("✅ Analysis successfully completed!")
        return results

    except Exception as e:
        # 记录分析失败的详细日志
        analysis_duration = time.time() - analysis_start_time

        logger_manager.log_module_error(
            logger, "comprehensive_analysis", stock_symbol, session_id,
            analysis_duration, str(e)
        )

        logger.error(f"❌ [分析失败] 股票分析执行失败",
                    extra={
                        'stock_symbol': stock_symbol,
                        'session_id': session_id,
                        'duration': analysis_duration,
                        'error': str(e),
                        'error_type': type(e).__name__,
                        'analysts_used': analysts,
                        'success': False,
                        'event_type': 'web_analysis_error'
                    }, exc_info=True)

        # 如果真实分析失败，返回错误信息而不是误导性演示数据
        return {
            'stock_symbol': stock_symbol,
            'analysis_date': analysis_date,
            'analysts': analysts,
            'research_depth': research_depth,
            'llm_provider': llm_provider,
            'llm_model': llm_model,
            'state': {},  # 空状态，将显示占位符
            'decision': {},  # 空决策
            'success': False,
            'error': str(e),
            'is_demo': False,
            'error_reason': f"分析失败: {str(e)}"
        }

def format_analysis_results(results):
    """格式化分析结果用于显示"""
    
    if not results['success']:
        return {
            'error': results['error'],
            'success': False
        }
    
    state = results['state']
    decision = results['decision']

    # 提取关键信息
    # decision 可能是字符串（如 "BUY", "SELL", "HOLD"）或字典
    if isinstance(decision, str):
        # Convert English investment advice to Chinese
        action_translation = {
            'BUY': '买入',
            'SELL': '卖出',
            'HOLD': '持有',
            'buy': '买入',
            'sell': '卖出',
            'hold': '持有'
        }
        action = action_translation.get(decision.strip(), decision.strip())

        formatted_decision = {
            'action': action,
            'confidence': 0.7,  # Default confidence
            'risk_score': 0.3,  # Default risk score
            'target_price': None,  # String format has no target price
            'reasoning': f'Based on AI analysis, recommend {decision.strip().upper()}'
        }
    elif isinstance(decision, dict):
        # Handle target price - ensure correct numerical extraction
        target_price = decision.get('target_price')
        if target_price is not None and target_price != 'N/A':
            try:
                # Try to convert to float
                if isinstance(target_price, str):
                    # Remove currency symbols and spaces
                    clean_price = target_price.replace('$', '').replace('¥', '').replace('￥', '').strip()
                    target_price = float(clean_price) if clean_price and clean_price != 'None' else None
                elif isinstance(target_price, (int, float)):
                    target_price = float(target_price)
                else:
                    target_price = None
            except (ValueError, TypeError):
                target_price = None
        else:
            target_price = None

        # Convert English investment advice to Chinese
        action_translation = {
            'BUY': '买入',
            'SELL': '卖出',
            'HOLD': '持有',
            'buy': '买入',
            'sell': '卖出',
            'hold': '持有'
        }
        action = decision.get('action', '持有')
        chinese_action = action_translation.get(action, action)

        formatted_decision = {
            'action': chinese_action,
            'confidence': decision.get('confidence', 0.5),
            'risk_score': decision.get('risk_score', 0.3),
            'target_price': target_price,
            'reasoning': decision.get('reasoning', 'No analysis reasoning available')
        }
    else:
        # Handle other types
        formatted_decision = {
            'action': '持有',
            'confidence': 0.5,
            'risk_score': 0.3,
            'target_price': None,
            'reasoning': f'Analysis result: {str(decision)}'
        }
    
    # 格式化状态信息
    formatted_state = {}
    
    # 处理各个分析模块的结果 - 包含完整的智能体团队分析
    analysis_keys = [
        'market_report',
        'fundamentals_report',
        'sentiment_report',
        'news_report',
        'risk_assessment',
        'investment_plan',
        # 添加缺失的团队决策数据，确保与CLI端一致
        'investment_debate_state',  # 研究团队辩论（多头/空头研究员）
        'trader_investment_plan',   # 交易团队计划
        'risk_debate_state',        # 风险管理团队决策
        'final_trade_decision'      # 最终交易决策
    ]
    
    for key in analysis_keys:
        if key in state:
            # 对文本内容进行中文化处理
            content = state[key]
            if isinstance(content, str):
                content = translate_analyst_labels(content)
            formatted_state[key] = content
        elif key == 'risk_assessment':
            # 特殊处理：从 risk_debate_state 生成 risk_assessment
            risk_assessment = extract_risk_assessment(state)
            if risk_assessment:
                formatted_state[key] = risk_assessment
    
    return {
        'stock_symbol': results['stock_symbol'],
        'decision': formatted_decision,
        'state': formatted_state,
        'success': True,
        # 将配置信息放在顶层，供前端直接访问
        'analysis_date': results['analysis_date'],
        'analysts': results['analysts'],
        'research_depth': results['research_depth'],
        'llm_provider': results.get('llm_provider', 'dashscope'),
        'llm_model': results['llm_model'],
        'metadata': {
            'analysis_date': results['analysis_date'],
            'analysts': results['analysts'],
            'research_depth': results['research_depth'],
            'llm_provider': results.get('llm_provider', 'dashscope'),
            'llm_model': results['llm_model']
        }
    }

def validate_analysis_params(stock_symbol, analysis_date, analysts, research_depth, market_type="US Stock"):
    """Validate analysis parameters"""

    errors = []

    # Validate stock symbol
    if not stock_symbol or len(stock_symbol.strip()) == 0:
        errors.append("Stock symbol cannot be empty")
    elif len(stock_symbol.strip()) > 10:
        errors.append("Stock symbol length cannot exceed 10 characters")
    else:
        # Validate code format based on market type
        symbol = stock_symbol.strip()
        if market_type == "A股":
            # A-share: 6 digits
            import re
            if not re.match(r'^\d{6}$', symbol):
                errors.append("A-share code format error, should be 6 digits (e.g.: 000001)")
        elif market_type == "港股":
            # H-share: 4-5 digits.HK or pure 4-5 digits
            import re
            symbol_upper = symbol.upper()
            # Check if it's XXXX.HK or XXXXX.HK format
            hk_format = re.match(r'^\d{4,5}\.HK$', symbol_upper)
            # Check if it's pure 4-5 digits format
            digit_format = re.match(r'^\d{4,5}$', symbol)

            if not (hk_format or digit_format):
                errors.append("H-share code format error, should be 4 digits.HK (e.g.: 0700.HK) or 4 digits (e.g.: 0700)")
        elif market_type == "US Stock":
            # US stock: 1-5 letters
            import re
            if not re.match(r'^[A-Z]{1,5}$', symbol.upper()):
                errors.append("US stock code format error, should be 1-5 letters (e.g.: AAPL)")
    
    # Validate analyst list
    if not analysts or len(analysts) == 0:
        errors.append("Must select at least one analyst")
    
    valid_analysts = ['market', 'social', 'news', 'fundamentals']
    invalid_analysts = [a for a in analysts if a not in valid_analysts]
    if invalid_analysts:
        errors.append(f"Invalid analyst types: {', '.join(invalid_analysts)}")
    
    # Validate research depth
    if not isinstance(research_depth, int) or research_depth < 1 or research_depth > 5:
        errors.append("Research depth must be an integer between 1-5")
    
    # Validate analysis date
    try:
        from datetime import datetime
        datetime.strptime(analysis_date, '%Y-%m-%d')
    except ValueError:
        errors.append("Invalid analysis date format, should be YYYY-MM-DD format")
    
    return len(errors) == 0, errors

def get_supported_stocks():
    """Get list of supported stocks"""
    
    # Common US stock symbols
    popular_stocks = [
        {'symbol': 'AAPL', 'name': 'Apple Inc.', 'sector': 'Technology'},
        {'symbol': 'MSFT', 'name': 'Microsoft', 'sector': 'Technology'},
        {'symbol': 'GOOGL', 'name': 'Google', 'sector': 'Technology'},
        {'symbol': 'AMZN', 'name': 'Amazon', 'sector': 'Consumer'},
        {'symbol': 'TSLA', 'name': 'Tesla', 'sector': 'Automotive'},
        {'symbol': 'NVDA', 'name': 'NVIDIA', 'sector': 'Technology'},
        {'symbol': 'META', 'name': 'Meta', 'sector': 'Technology'},
        {'symbol': 'NFLX', 'name': 'Netflix', 'sector': 'Media'},
        {'symbol': 'AMD', 'name': 'AMD', 'sector': 'Technology'},
        {'symbol': 'INTC', 'name': 'Intel', 'sector': 'Technology'},
        {'symbol': 'SPY', 'name': 'S&P 500 ETF', 'sector': 'ETF'},
        {'symbol': 'QQQ', 'name': 'Nasdaq 100 ETF', 'sector': 'ETF'},
    ]
    
    return popular_stocks

def generate_demo_results_deprecated(stock_symbol, analysis_date, analysts, research_depth, llm_provider, llm_model, error_msg, market_type="US Stock"):
    """
    Deprecated: Generate demo analysis results

    Note: This function is deprecated because demo data can mislead users.
    We now use placeholders instead of demo data.
    """

    import random

    # Set currency symbol and price range based on market type
    if market_type == "HK Stock":
        currency_symbol = "HK$"
        price_range = (50, 500)  # HK stock price range
        market_name = "HK Stock"
    elif market_type == "A Stock":
        currency_symbol = "¥"
        price_range = (5, 100)   # A stock price range
        market_name = "A Stock"
    else:  # US Stock
        currency_symbol = "$"
        price_range = (50, 300)  # US stock price range
        market_name = "US Stock"

    # Generate simulated decision
    actions = ['BUY', 'HOLD', 'SELL']
    action = random.choice(actions)

    demo_decision = {
        'action': action,
        'confidence': round(random.uniform(0.6, 0.9), 2),
        'risk_score': round(random.uniform(0.2, 0.7), 2),
        'target_price': round(random.uniform(*price_range), 2),
        'reasoning': f"""
Based on comprehensive analysis of {market_name} {stock_symbol}, our AI analysis team concludes:

**Investment Recommendation**: {action}
**Target Price**: {currency_symbol}{round(random.uniform(*price_range), 2)}

**Key Analysis Points**:
1. **Technical Analysis**: Current price trend shows {'upward' if action == 'BUY' else 'downward' if action == 'SELL' else 'sideways'} signals
2. **Fundamental Assessment**: Company financial condition is {'good' if action == 'BUY' else 'average' if action == 'HOLD' else 'needs attention'}
3. **Market Sentiment**: Investor sentiment is {'optimistic' if action == 'BUY' else 'neutral' if action == 'HOLD' else 'cautious'}
4. **Risk Assessment**: Current risk level is {'moderate' if action == 'HOLD' else 'low' if action == 'BUY' else 'high'}

**Note**: This is demo data, actual analysis requires proper API key configuration.
        """
    }

    # Generate simulated state data
    demo_state = {}

    if 'market' in analysts:
        current_price = round(random.uniform(*price_range), 2)
        high_price = round(current_price * random.uniform(1.2, 1.8), 2)
        low_price = round(current_price * random.uniform(0.5, 0.8), 2)

        demo_state['market_report'] = f"""
## 📈 {market_name}{stock_symbol} 技术面分析报告

### 价格趋势分析
- **当前价格**: {currency_symbol}{current_price}
- **日内变化**: {random.choice(['+', '-'])}{round(random.uniform(0.5, 5), 2)}%
- **52周高点**: {currency_symbol}{high_price}
- **52周低点**: {currency_symbol}{low_price}

### 技术指标
- **RSI (14日)**: {round(random.uniform(30, 70), 1)}
- **MACD**: {'看涨' if action == 'BUY' else '看跌' if action == 'SELL' else '中性'}
- **移动平均线**: 价格{'高于' if action == 'BUY' else '低于' if action == 'SELL' else '接近'}20日均线

### 支撑阻力位
- **支撑位**: ${round(random.uniform(80, 120), 2)}
- **阻力位**: ${round(random.uniform(250, 350), 2)}

*注意: 这是演示数据，实际分析需要配置API密钥*
        """

    if 'fundamentals' in analysts:
        demo_state['fundamentals_report'] = f"""
## 💰 {stock_symbol} 基本面分析报告

### 财务指标
- **市盈率 (P/E)**: {round(random.uniform(15, 35), 1)}
- **市净率 (P/B)**: {round(random.uniform(1, 5), 1)}
- **净资产收益率 (ROE)**: {round(random.uniform(10, 25), 1)}%
- **毛利率**: {round(random.uniform(20, 60), 1)}%

### 盈利能力
- **营收增长**: {random.choice(['+', '-'])}{round(random.uniform(5, 20), 1)}%
- **净利润增长**: {random.choice(['+', '-'])}{round(random.uniform(10, 30), 1)}%
- **每股收益**: ${round(random.uniform(2, 15), 2)}

### 财务健康度
- **负债率**: {round(random.uniform(20, 60), 1)}%
- **流动比率**: {round(random.uniform(1, 3), 1)}
- **现金流**: {'正向' if action != 'SELL' else '需关注'}

*注意: 这是演示数据，实际分析需要配置API密钥*
        """

    if 'social' in analysts:
        demo_state['sentiment_report'] = f"""
## 💭 {stock_symbol} 市场情绪分析报告

### 社交媒体情绪
- **整体情绪**: {'积极' if action == 'BUY' else '消极' if action == 'SELL' else '中性'}
- **情绪强度**: {round(random.uniform(0.5, 0.9), 2)}
- **讨论热度**: {'高' if random.random() > 0.5 else '中等'}

### 投资者情绪指标
- **恐慌贪婪指数**: {round(random.uniform(20, 80), 0)}
- **看涨看跌比**: {round(random.uniform(0.8, 1.5), 2)}
- **期权Put/Call比**: {round(random.uniform(0.5, 1.2), 2)}

### 机构投资者动向
- **机构持仓变化**: {random.choice(['增持', '减持', '维持'])}
- **分析师评级**: {'买入' if action == 'BUY' else '卖出' if action == 'SELL' else '持有'}

*注意: 这是演示数据，实际分析需要配置API密钥*
        """

    if 'news' in analysts:
        demo_state['news_report'] = f"""
## 📰 {stock_symbol} 新闻事件分析报告

### 近期重要新闻
1. **财报发布**: 公司发布{'超预期' if action == 'BUY' else '低于预期' if action == 'SELL' else '符合预期'}的季度财报
2. **行业动态**: 所在行业面临{'利好' if action == 'BUY' else '挑战' if action == 'SELL' else '稳定'}政策环境
3. **公司公告**: 管理层{'乐观' if action == 'BUY' else '谨慎' if action == 'SELL' else '稳健'}展望未来

### 新闻情绪分析
- **正面新闻占比**: {round(random.uniform(40, 80), 0)}%
- **负面新闻占比**: {round(random.uniform(10, 40), 0)}%
- **中性新闻占比**: {round(random.uniform(20, 50), 0)}%

### 市场影响评估
- **短期影响**: {'正面' if action == 'BUY' else '负面' if action == 'SELL' else '中性'}
- **长期影响**: {'积极' if action != 'SELL' else '需观察'}

*注意: 这是演示数据，实际分析需要配置API密钥*
        """

    # 添加风险评估和投资建议
    demo_state['risk_assessment'] = f"""
## ⚠️ {stock_symbol} 风险评估报告

### 主要风险因素
1. **市场风险**: {'低' if action == 'BUY' else '高' if action == 'SELL' else '中等'}
2. **行业风险**: {'可控' if action != 'SELL' else '需关注'}
3. **公司特定风险**: {'较低' if action == 'BUY' else '中等'}

### 风险等级评估
- **总体风险等级**: {'低风险' if action == 'BUY' else '高风险' if action == 'SELL' else '中等风险'}
- **建议仓位**: {random.choice(['轻仓', '标准仓位', '重仓']) if action != 'SELL' else '建议减仓'}

*注意: 这是演示数据，实际分析需要配置API密钥*
    """

    demo_state['investment_plan'] = f"""
## 📋 {stock_symbol} 投资建议

### 具体操作建议
- **操作方向**: {action}
- **建议价位**: ${round(random.uniform(90, 310), 2)}
- **止损位**: ${round(random.uniform(80, 200), 2)}
- **目标价位**: ${round(random.uniform(150, 400), 2)}

### 投资策略
- **投资期限**: {'短期' if research_depth <= 2 else '中长期'}
- **仓位管理**: {'分批建仓' if action == 'BUY' else '分批减仓' if action == 'SELL' else '维持现状'}

*注意: 这是演示数据，实际分析需要配置API密钥*
    """

    # 添加团队决策演示数据，确保与CLI端一致
    demo_state['investment_debate_state'] = {
        'bull_history': f"""
## 📈 多头研究员分析

作为多头研究员，我对{stock_symbol}持乐观态度：

### 🚀 投资亮点
1. **技术面突破**: 股价突破关键阻力位，技术形态良好
2. **基本面支撑**: 公司业绩稳健增长，财务状况健康
3. **市场机会**: 当前估值合理，具备上涨空间

### 📊 数据支持
- 近期成交量放大，资金流入明显
- 行业景气度提升，政策环境有利
- 机构投资者增持，市场信心增强

**建议**: 积极买入，目标价位上调15-20%

*注意: 这是演示数据*
        """.strip(),

        'bear_history': f"""
## 📉 空头研究员分析

作为空头研究员，我对{stock_symbol}持谨慎态度：

### ⚠️ 风险因素
1. **估值偏高**: 当前市盈率超过行业平均水平
2. **技术风险**: 短期涨幅过大，存在回调压力
3. **宏观环境**: 市场整体波动加大，不确定性增加

### 📉 担忧点
- 成交量虽然放大，但可能是获利盘出货
- 行业竞争加剧，公司市场份额面临挑战
- 政策变化可能对行业产生负面影响

**建议**: 谨慎观望，等待更好的入场时机

*注意: 这是演示数据*
        """.strip(),

        'judge_decision': f"""
## 🎯 研究经理综合决策

经过aeda和空头研究员的充分辩论，我的综合判断如下：

### 📊 综合评估
- **多头观点**: 技术面和基本面都显示积极信号
- **空头观点**: 估值和短期风险需要关注
- **平衡考虑**: 机会与风险并存，需要策略性操作

### 🎯 最终建议
基于当前市场环境和{stock_symbol}的具体情况，建议采取**{action}**策略：

1. **操作建议**: {action}
2. **仓位控制**: {'分批建仓' if action == '买入' else '分批减仓' if action == '卖出' else '维持现状'}
3. **风险管理**: 设置止损位，控制单只股票仓位不超过10%

**决策依据**: 综合技术面、基本面和市场情绪分析

*注意: 这是演示数据*
        """.strip()
    }

    demo_state['trader_investment_plan'] = f"""
## 💼 交易团队执行计划

基于研究团队的分析结果，制定如下交易执行计划：

### 🎯 交易策略
- **交易方向**: {action}
- **目标价位**: {currency_symbol}{round(random.uniform(*price_range) * 1.1, 2)}
- **止损价位**: {currency_symbol}{round(random.uniform(*price_range) * 0.9, 2)}

### 📊 仓位管理
- **建议仓位**: {'30-50%' if action == '买入' else '减仓至20%' if action == '卖出' else '维持现有仓位'}
- **分批操作**: {'分3次建仓' if action == '买入' else '分2次减仓' if action == '卖出' else '暂不操作'}
- **时间安排**: {'1-2周内完成' if action != '持有' else '持续观察'}

### ⚠️ 风险控制
- **止损设置**: 跌破支撑位立即止损
- **止盈策略**: 达到目标价位分批止盈
- **监控要点**: 密切关注成交量和技术指标变化

*注意: 这是演示数据，实际交易需要配置API密钥*
    """

    demo_state['risk_debate_state'] = {
        'risky_history': f"""
## 🚀 激进分析师风险评估

从激进投资角度分析{stock_symbol}：

### 💪 风险承受能力
- **高收益机会**: 当前市场提供了难得的投资机会
- **风险可控**: 虽然存在波动，但长期趋势向好
- **时机把握**: 现在是积极布局的最佳时机

### 🎯 激进策略
- **加大仓位**: 建议将仓位提升至60-80%
- **杠杆使用**: 可适度使用杠杆放大收益
- **快速行动**: 机会稍纵即逝，需要果断决策

**风险评级**: 中等风险，高收益潜力

*注意: 这是演示数据*
        """.strip(),

        'safe_history': f"""
## 🛡️ 保守分析师风险评估

从风险控制角度分析{stock_symbol}：

### ⚠️ 风险识别
- **市场波动**: 当前市场不确定高
- **估值风险**: 部分股票估值已经偏高
- **流动性风险**: 需要关注市场流动性变化

### 🔒 保守策略
- **控制仓位**: 建议仓位不超过30%
- **分散投资**: 避免过度集中于单一标的
- **安全边际**: 确保有足够的安全边际

**风险评级**: 中高风险，需要谨慎操作

*注意: 这是演示数据*
        """.strip(),

        'neutral_history': f"""
## ⚖️ 中性分析师风险评估

从平衡角度分析{stock_symbol}：

### 📊 客观评估
- **机会与风险并存**: 当前市场既有机会也有风险
- **适度参与**: 建议采取适度参与的策略
- **灵活调整**: 根据市场变化及时调整策略

### ⚖️ 平衡策略
- **中等仓位**: 建议仓位控制在40-50%
- **动态调整**: 根据市场情况动态调整仓位
- **风险监控**: 持续监控风险指标变化

**风险评级**: 中等风险，平衡收益

*注意: 这是演示数据*
        """.strip(),

        'judge_decision': f"""
## 🎯 投资组合经理最终风险决策

综合三位风险分析师的意见，最终风险管理决策如下：

### 📊 风险综合评估
- **激进观点**: 高收益机会，建议积极参与
- **保守观点**: 风险较高，建议谨慎操作
- **中性观点**: 机会与风险并存，适度参与

### 🎯 最终风险决策
基于当前市场环境和{stock_symbol}的风险特征：

1. **风险等级**: 中等风险
2. **建议仓位**: 40%（平衡收益与风险）
3. **风险控制**: 严格执行止损策略
4. **监控频率**: 每日监控，及时调整

**决策理由**: 在控制风险的前提下，适度参与市场机会

*注意: 这是演示数据*
        """.strip()
    }

    demo_state['final_trade_decision'] = f"""
## 🎯 最终投资决策

经过分析师团队、研究团队、交易团队和风险管理团队的全面分析，最终投资决策如下：

### 📊 决策摘要
- **投资建议**: **{action}**
- **置信度**: {confidence:.1%}
- **风险评级**: 中等风险
- **预期收益**: {'10-20%' if action == '买入' else '规避损失' if action == '卖出' else '稳健持有'}

### 🎯 执行计划
1. **操作方向**: {action}
2. **目标仓位**: {'40%' if action == '买入' else '20%' if action == '卖出' else '维持现状'}
3. **执行时间**: {'1-2周内分批执行' if action != '持有' else '持续观察'}
4. **风险控制**: 严格执行止损止盈策略

### 📈 预期目标
- **目标价位**: {currency_symbol}{round(random.uniform(*price_range) * 1.15, 2)}
- **止损价位**: {currency_symbol}{round(random.uniform(*price_range) * 0.85, 2)}
- **投资期限**: {'3-6个月' if research_depth >= 3 else '1-3个月'}

### ⚠️ 重要提醒
这是基于当前市场环境和{stock_symbol}基本面的综合判断。投资有风险，请根据个人风险承受能力谨慎决策。

**免责声明**: 本分析仅供参考，不构成投资建议。

*注意: 这是演示数据，实际分析需要配置正确的API密钥*
    """

    return {
        'stock_symbol': stock_symbol,
        'analysis_date': analysis_date,
        'analysts': analysts,
        'research_depth': research_depth,
        'llm_provider': llm_provider,
        'llm_model': llm_model,
        'state': demo_state,
        'decision': demo_decision,
        'success': True,
        'error': None,
        'is_demo': True,
        'demo_reason': f"API调用失败，显示演示数据。错误信息: {error_msg}"
    }
