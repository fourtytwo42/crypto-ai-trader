#!/usr/bin/env python3
"""Show example token data from the database."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from pumpfun_train.db import get_pumpfun_db_manager
from pumpfun_train.models import PumpToken, PumpTrade, PumpCandle1m, PumpFeature1m
from sqlalchemy import func, select
from datetime import datetime

def format_timestamp(ts):
    if ts is None:
        return None
    if isinstance(ts, datetime):
        return ts.isoformat()
    # Assume milliseconds
    return datetime.fromtimestamp(ts / 1000).isoformat()

def main():
    db = get_pumpfun_db_manager()
    with db.session() as session:
        # Get a token with lots of data as example
        result = session.execute(
            select(
                PumpToken.id,
                PumpToken.mint_address,
                PumpToken.symbol,
                PumpToken.name,
                PumpToken.created_timestamp,
                PumpToken.king_of_the_hill_timestamp,
                PumpToken.completed,
                func.count(PumpTrade.id.distinct()).label('total_trades'),
                func.min(PumpTrade.timestamp).label('first_trade_ts'),
                func.max(PumpTrade.timestamp).label('last_trade_ts'),
                func.count(PumpCandle1m.id.distinct()).label('total_candles'),
                func.count(PumpFeature1m.id.distinct()).label('total_features'),
            )
            .outerjoin(PumpTrade, PumpTrade.token_id == PumpToken.id)
            .outerjoin(PumpCandle1m, PumpCandle1m.token_id == PumpToken.id)
            .outerjoin(PumpFeature1m, PumpFeature1m.token_id == PumpToken.id)
            .group_by(PumpToken.id)
            .having(func.count(PumpTrade.id) > 100)  # Token with significant trades
            .order_by(func.count(PumpTrade.id).desc())
            .limit(1)
        ).first()
        
        if not result:
            print("No tokens found with sufficient data")
            return
        
        token_id = result.id
        
        print("=" * 80)
        print("PUMP.FUN TOKEN DATA BREAKDOWN")
        print("=" * 80)
        print()
        
        print("=" * 80)
        print("1. TOKEN METADATA (tokens table)")
        print("=" * 80)
        print(f"ID:                           {result.id}")
        print(f"Mint Address:                 {result.mint_address}")
        print(f"Symbol:                       {result.symbol}")
        print(f"Name:                         {result.name}")
        print(f"Created Timestamp (ms):       {result.created_timestamp}")
        print(f"Created Timestamp (readable): {format_timestamp(result.created_timestamp)}")
        print(f"King of the Hill Timestamp:   {result.king_of_the_hill_timestamp}")
        print(f"King of the Hill (readable):  {format_timestamp(result.king_of_the_hill_timestamp)}")
        print(f"Completed (bonded out):       {result.completed}")
        print(f"Total Trades:                 {result.total_trades:,}")
        print(f"First Trade Timestamp:        {format_timestamp(result.first_trade_ts)}")
        print(f"Last Trade Timestamp:         {format_timestamp(result.last_trade_ts)}")
        print(f"Total Candles (1m):           {result.total_candles:,}")
        print(f"Total Features (1m):          {result.total_features:,}")
        print()
        
        # Get sample trades
        sample_trades = session.execute(
            select(PumpTrade)
            .where(PumpTrade.token_id == token_id)
            .order_by(PumpTrade.timestamp)
            .limit(3)
        ).all()
        
        print("=" * 80)
        print("2. TRADE DATA (trades table) - Sample of first 3 trades")
        print("=" * 80)
        print("Each trade represents an individual buy/sell transaction on pump.fun")
        print()
        for i, trade in enumerate(sample_trades, 1):
            print(f"Trade #{i} (ID: {trade.id}):")
            print(f"  Timestamp (ms):     {trade.timestamp}")
            print(f"  Timestamp (readable): {format_timestamp(trade.timestamp)}")
            print(f"  Price SOL:          {trade.price_sol}")
            print(f"  Price USD:          {trade.price_usd}")
            print(f"  Amount SOL:         {trade.amount_sol}")
            print(f"  Amount USD:         {trade.amount_usd}")
            print(f"  Base Amount:        {trade.base_amount}")
            print()
        
        # Get trade stats
        trade_stats = session.execute(
            select(
                func.count(PumpTrade.id).label('count'),
                func.min(PumpTrade.timestamp).label('first'),
                func.max(PumpTrade.timestamp).label('last'),
                func.avg(PumpTrade.price_usd).label('avg_price'),
                func.sum(PumpTrade.amount_usd).label('total_volume'),
            )
            .where(PumpTrade.token_id == token_id)
        ).first()
        
        print(f"Trade Statistics for this token:")
        print(f"  Total Trades:        {trade_stats.count:,}")
        print(f"  First Trade:         {format_timestamp(trade_stats.first)}")
        print(f"  Last Trade:          {format_timestamp(trade_stats.last)}")
        print(f"  Average Price USD:   ${trade_stats.avg_price:,.8f}" if trade_stats.avg_price else "  Average Price USD:   N/A")
        print(f"  Total Volume USD:    ${trade_stats.total_volume:,.2f}" if trade_stats.total_volume else "  Total Volume USD:    N/A")
        print()
        
        # Get sample candles
        sample_candles = session.execute(
            select(PumpCandle1m)
            .where(PumpCandle1m.token_id == token_id)
            .order_by(PumpCandle1m.timestamp)
            .limit(3)
        ).all()
        
        print("=" * 80)
        print("3. CANDLE DATA (pump_candles_1m table) - Sample of first 3 candles")
        print("=" * 80)
        print("1-minute OHLCV candles derived from trades")
        print()
        for i, candle in enumerate(sample_candles, 1):
            print(f"Candle #{i} (Timestamp: {candle.timestamp.isoformat()}):")
            print(f"  Open:               ${candle.open:,.8f}")
            print(f"  High:               ${candle.high:,.8f}")
            print(f"  Low:                ${candle.low:,.8f}")
            print(f"  Close:              ${candle.close:,.8f}")
            print(f"  Volume USD:         ${candle.volume_usd:,.2f}")
            print(f"  Volume SOL:         {candle.volume_sol:,.8f}")
            print(f"  Number of Trades:   {candle.trades}")
            print(f"  Created At:         {candle.created_at.isoformat()}")
            print()
        
        # Get sample features
        sample_features = session.execute(
            select(PumpFeature1m)
            .where(PumpFeature1m.token_id == token_id)
            .order_by(PumpFeature1m.timestamp)
            .limit(3)
        ).all()
        
        print("=" * 80)
        print("4. FEATURE DATA (pump_features_1m table) - Sample of first 3 features")
        print("=" * 80)
        print("Technical indicators derived from candles (used for model training)")
        print()
        for i, feature in enumerate(sample_features, 1):
            print(f"Feature #{i} (Timestamp: {feature.timestamp.isoformat()}):")
            print(f"  Return:             {feature.return_}")
            print(f"  Range:              {feature.range}")
            print(f"  Body:               {feature.body}")
            print(f"  DLog Volume:        {feature.dlog_volume}")
            print(f"  Ret Mean 15:        {feature.ret_mean_15}")
            print(f"  Ret Std 15:         {feature.ret_std_15}")
            print(f"  Ret Mean 60:        {feature.ret_mean_60}")
            print(f"  Ret Std 60:         {feature.ret_std_60}")
            print(f"  Created At:         {feature.created_at.isoformat()}")
            print()
        
        print("=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print()
        print("Database Tables:")
        print("  - tokens:              Token metadata (ID, symbol, name, timestamps)")
        print("  - trades:              Raw trade stream (price, amount, timestamp)")
        print("  - pump_candles_1m:     Derived 1-minute OHLCV candles")
        print("  - pump_features_1m:    Derived technical indicators for training")
        print("  - pump_sol_prices:     Cached SOL/USD prices for normalization")
        print()
        print("Data Flow:")
        print("  trades → pump_candles_1m → pump_features_1m → model training")
        print()
        print("All timestamp fields use milliseconds since epoch (BigInteger)")
        print("Datetime fields in candles/features use UTC timezone-aware datetimes")

if __name__ == "__main__":
    main()

