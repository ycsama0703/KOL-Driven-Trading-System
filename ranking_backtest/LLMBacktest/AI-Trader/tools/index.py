import json
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# --- 辅助函数 ---

def _calculate_exponential_moving_average(data_series: pd.Series, period_length: int) -> pd.Series:
    """计算指数移动平均线 (Exponential Moving Average, EMA)"""
    return data_series.ewm(span=period_length, adjust=False).mean()

# --- 指标计算函数 ---

def calculate_moving_average(
    price_data: pd.DataFrame, period_length: int = 20, close_price_column: str = "4. sell price"
) -> pd.Series:
    """
    用收盘价("4. sell price")计算移动平均指标 (Moving Average, MA)。
    这里使用指数移动平均线 (EMA) 作为实现。
    """
    if price_data.empty or close_price_column not in price_data.columns:
        return pd.Series(dtype=float)
    
    # 使用指数移动平均线 (EMA)
    return _calculate_exponential_moving_average(price_data[close_price_column], period_length)

def calculate_macd(
    price_data: pd.DataFrame, 
    fast_ema_period: int = 12, 
    slow_ema_period: int = 26, 
    signal_ema_period: int = 9, 
    close_price_column: str = "4. sell price"
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    用收盘价("4. sell price")计算MACD指标 (Moving Average Convergence Divergence)。
    返回 (MACD线, 信号线, MACD柱)
    """
    if price_data.empty or close_price_column not in price_data.columns:
        empty_series = pd.Series(dtype=float)
        return empty_series, empty_series, empty_series

    closing_prices = price_data[close_price_column]
    
    # 1. 计算快线EMA (Fast EMA) 和慢线EMA (Slow EMA)
    ema_fast = _calculate_exponential_moving_average(closing_prices, fast_ema_period)
    ema_slow = _calculate_exponential_moving_average(closing_prices, slow_ema_period)
    
    # 2. 计算MACD线
    macd_line = ema_fast - ema_slow
    
    # 3. 计算信号线 (Signal Line, MACD线的EMA)
    signal_line = _calculate_exponential_moving_average(macd_line, signal_ema_period)
    
    # 4. 计算MACD柱 (MACD Histogram)
    macd_histogram = macd_line - signal_line
    
    return macd_line, signal_line, macd_histogram

def calculate_rsi(
    price_data: pd.DataFrame, period_length: int = 14, close_price_column: str = "4. sell price"
) -> pd.Series:
    """
    用收盘价("4. sell price")计算RSI指标 (Relative Strength Index)
    """
    if price_data.empty or close_price_column not in price_data.columns:
        return pd.Series(dtype=float)

    # 计算价格变化
    price_change = price_data[close_price_column].diff()
    
    # 分离上涨和下跌
    gain = price_change.where(price_change > 0, 0) # 上涨值
    loss = -price_change.where(price_change < 0, 0) # 下跌值的绝对值
    
    # 使用 EMA 平滑计算平均收益和平均损失
    average_gain = gain.ewm(span=period_length, adjust=False).mean()
    average_loss = loss.ewm(span=period_length, adjust=False).mean()
    
    # 计算相对强度 (Relative Strength, RS)
    # 避免除以零
    relative_strength = average_gain / average_loss.replace(0, np.nan) 
    
    # 计算RSI
    relative_strength_index = 100 - (100 / (1 + relative_strength))
    
    return relative_strength_index

def calculate_cmf(
    price_data: pd.DataFrame, 
    period_length: int = 20,
    high_price_column: str = "2. high",
    low_price_column: str = "3. low",
    close_price_column: str = "4. sell price",
    volume_column: str = "5. volume"
) -> pd.Series:
    """
    计算 Chaikin Money Flow (CMF) 成交量趋势指标。
    """
    required_cols = [high_price_column, low_price_column, close_price_column, volume_column]
    if price_data.empty or not all(col in price_data.columns for col in required_cols):
        return pd.Series(dtype=float)

    high = price_data[high_price_column]
    low = price_data[low_price_column]
    close = price_data[close_price_column]
    volume = price_data[volume_column]
    
    # 1. 资金流量乘数 (Money Flow Multiplier)
    # 避免除以零
    high_minus_low = high - low
    mfm = ((close - low) - (high - close)) / high_minus_low.replace(0, np.nan)
    
    # 2. 资金流量额 (Money Flow Volume)
    mfv = mfm * volume
    
    # 3. CMF = 周期内 MFV 之和 / 周期内 Volume 之和
    # 在分母为零时，结果可能为 NaN
    cmf = mfv.rolling(window=period_length).sum() / volume.rolling(window=period_length).sum().replace(0, np.nan)
    
    return cmf

def calculate_trend_reference(
    price_data: pd.DataFrame, 
    high_price_column: str = "2. high", 
    low_price_column: str = "3. low"
) -> Dict[str, Any]:
    """
    用最高价("2. high")/最低价("3. low")计算趋势线的简单参考。
    这里简单计算过去60天的最高价/最低价，作为趋势强弱的简单参考。
    """
    if price_data.empty or high_price_column not in price_data.columns or low_price_column not in price_data.columns:
        return {"max_high_60d": None, "min_low_60d": None}

    # 简单地计算过去60天的最高价和最低价
    max_high_60d = price_data[high_price_column].max()
    min_low_60d = price_data[low_price_column].min()

    return {
        "max_high_60d": max_high_60d,
        "min_low_60d": min_low_60d
    }

# --- 主数据获取函数 ---

def get_technical_indicators(
    current_date_str: str, 
    stock_symbols: List[str], 
    merged_data_path: Optional[str] = None, 
    market_type: str = "us"
) -> Dict[str, Dict[str, float]]:
    """
    从数据文件 (merged.jsonl) 中读取一系列过去60天内价格；并计算各项指标。
    **所有计算出的指标和价格值都将四舍五入到两位小数。**

    Args:
        current_date_str: 日期字符串，格式 YYYY-MM-DD，代表今天日期。
        stock_symbols: 需要查询的股票代码列表。
        merged_data_path: 可选，自定义 merged.jsonl 路径；默认读取项目根目录下 data/merged.jsonl。
        market_type: 市场类型，"us" 为美股，"cn" 为A股

    """
    indicator_results: Dict[str, Dict[str, float]] = {symbol: {} for symbol in stock_symbols}
    wanted_symbols = set(stock_symbols)
    
    # 计算60天前的日期作为数据筛选的阈值
    try:
        current_date = datetime.strptime(current_date_str, "%Y-%m-%d").date()
        threshold_date = current_date - timedelta(days=60)
        date_threshold = threshold_date.strftime("%Y-%m-%d")
    except ValueError:
        print(f"Error: Invalid date format for current_date_str: {current_date_str}")
        return indicator_results

    # 确定 merged.jsonl 文件路径
    if merged_data_path:
        merged_file_path = Path(merged_data_path)
    else:
        # 假设文件在当前脚本所在目录的上上级目录的 data 文件夹下
        base_directory = Path(__file__).resolve().parents[1]
        merged_file_path = base_directory / "data" / "merged.jsonl"

    if not merged_file_path.exists():
        print(f"Warning: merged file not found at {merged_file_path}")
        return indicator_results

    # 用于存储每个标的历史数据
    all_symbol_history_data: Dict[str, Dict[str, Dict[str, str]]] = {symbol: {} for symbol in stock_symbols}

    with merged_file_path.open("r", encoding="utf-8") as file_handle:
        for line_content in file_handle:
            if not line_content.strip():
                continue
            try:
                document = json.loads(line_content)
            except Exception as e:
                print(f"Error loading JSON line: {e}")
                continue
            
            meta_data = document.get("Meta Data", {}) if isinstance(document, dict) else {}
            symbol = meta_data.get("2. Symbol")
            if symbol not in wanted_symbols:
                continue
            
            # 查找以 "Time Series" 开头的键（例如 "Time Series (Daily)"）
            time_series_data: Optional[Dict[str, Dict[str, str]]] = None
            for key, value in document.items():
                if key.startswith("Time Series") and isinstance(value, dict):
                    time_series_data = value
                    break
            
            if not time_series_data:
                continue
            
            # 筛选出过去60天的数据
            filtered_history_data = {
                date: price_values for date, price_values in time_series_data.items() if date >= date_threshold and date <= current_date_str
            }
            
            if filtered_history_data:
                all_symbol_history_data[symbol] = filtered_history_data

    # 计算指标并存储结果
    for symbol, raw_data_dict in all_symbol_history_data.items():
        if not raw_data_dict:
            continue
            
        # 转换数据为 Pandas DataFrame，并按日期升序排序
        data_frame = pd.DataFrame.from_dict(raw_data_dict, orient='index')
        data_frame.index.name = 'Date'
        
        # 确保列是数值类型
        for column_name in data_frame.columns:
            data_frame[column_name] = pd.to_numeric(data_frame[column_name], errors='coerce')
        
        # 按日期排序，确保时间序列计算的正确性
        data_frame = data_frame.sort_index(ascending=True)

        
        
        if data_frame.empty:
            continue
            
        # --- 计算指标 ---         

        # 1. MACD
        macd_line_series, signal_line_series, macd_histogram_series = calculate_macd(data_frame)
        
        # 2. RSI
        rsi_series = calculate_rsi(data_frame)
        
        # 3. MA (20日EMA)
        ma_20d_series = calculate_moving_average(data_frame, period_length=20)
        
        # 4. Trend (60日高低)
        trend_info = calculate_trend_reference(data_frame)
        
        # 5. CMF (成交量趋势指标)
        cmf_series = calculate_cmf(data_frame) # 使用默认20日周期

        # 获取最新的指标值 (即DataFrame/Series的最后一行/值)
        
        # MACD (最新值) - 两位小数
        if not macd_line_series.empty:
            indicator_results[symbol]["MACD_Line"] = round(macd_line_series.iloc[-1], 2)
            indicator_results[symbol]["Signal_Line"] = round(signal_line_series.iloc[-1], 2)
            indicator_results[symbol]["MACD_Hist"] = round(macd_histogram_series.iloc[-1], 2)
            
        # RSI (最新值) - 两位小数
        if not rsi_series.empty:
            indicator_results[symbol]["RSI"] = round(rsi_series.iloc[-1], 2)
            
        # MA (最新值) - 两位小数
        if not ma_20d_series.empty:
            indicator_results[symbol]["MA_20D"] = round(ma_20d_series.iloc[-1], 2)
            
        # CMF (最新值) - 两位小数
        if not cmf_series.empty:
            indicator_results[symbol]["CMF_20D"] = round(cmf_series.iloc[-1], 2)
            
        # Trend (60日高低) - 两位小数
        if trend_info["max_high_60d"] is not None:
             indicator_results[symbol]["60D_High"] = round(trend_info["max_high_60d"], 2)
        if trend_info["min_low_60d"] is not None:
             indicator_results[symbol]["60D_Low"] = round(trend_info["min_low_60d"], 2)

        # 最新买入价 - 两位小数
        if "1. buy price" in data_frame.columns and not data_frame.empty and not pd.isna(data_frame.iloc[-1]["1. buy price"]):
             indicator_results[symbol]["current_buying_price"] = round(data_frame.iloc[-1]["1. buy price"], 2)
        
    return indicator_results

def format_indicator_results(results_dict: Dict[str, Dict[str, float]]) -> str:
    lines = []
    for symbol, indicators in results_dict.items():
        lines.append(f"#### {symbol}")
        for indicator_name, value in indicators.items():
            # 使用 f-string 格式化输出，确保始终显示两位小数，即使结果是整数
            # 这是一个展示性的格式化，实际的数值已经通过 round() 确定
            if isinstance(value, (int, float)):
                 lines.append(f"{indicator_name}: {value:.2f}")
            else:
                 lines.append(f"{indicator_name}: {value}")
        lines.append("") # 空行分隔
    return "\n".join(lines)

def get_stock_symbols():
    # 假设该路径存在并包含股票代码列表
    symbols_file_path = "./data/symbols.csv"
    try:
        with open(symbols_file_path, encoding="utf-8") as file_handle:
            stock_symbols_list = [line.strip() for line in file_handle]
        return stock_symbols_list
    except FileNotFoundError:
        print(f"Error: symbols file not found at {symbols_file_path}")
        return []

def get_format_results(current_date, stock_symbols, merged_data_path):
    results = get_technical_indicators(current_date, stock_symbols, merged_data_path)
    
    return format_indicator_results(results)

