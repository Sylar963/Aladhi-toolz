#!/usr/bin/env python3
"""
Enhanced MAD Straddle Analyzer
==============================

This module integrates the enhanced volatility models with the existing MAD analysis system.
Provides a complete solution that combines:
- Original MAD-based tail risk analysis
- Forward volatility expectations (GARCH + regime awareness)
- Dynamic volatility surface modeling
- Enhanced straddle pricing

Addresses the core question: "Are straddles truly underpriced when accounting for volatility dynamics?"
"""

import asyncio
import json
import logging
import math
import time
import statistics
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from collections import deque
from dataclasses import dataclass

# Configure matplotlib backend
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import thalex_py.thalex as th
import keys

# Import existing MAD analysis components
from MADStraddle import MADStraddleAnalyzer, ExpirationData, SimpleOption, MADAnalysis
from StraddlePositionCalculator import InteractiveStraddleSelector, StraddlePositionCalculator

# Import new volatility modeling components
from volatility_regime import VolatilityRegimeDetector, VolatilityRegime
from forward_volatility import ForwardVolatilityEstimator, VolatilityForecast
from volatility_surface import VolatilitySurface
from enhanced_straddle_pricing import EnhancedStraddlePricingModel, EnhancedStraddlePricing, StraddleBuyingSignal
from financial_math import FinancialMath, TradingConstants

# Import new persistent data management components
from supabase_data_manager import SupabaseDataManager, VolatilityAnalysisData
from supabase_config import create_supabase_config


@dataclass
class IntegratedStraddleAnalysis:
    """Complete straddle analysis combining MAD and enhanced volatility models"""
    
    # Basic option information
    expiry_date: str
    strike_price: float
    time_to_expiry: float
    straddle_price: float
    
    # Original MAD analysis
    mad_analysis: MADAnalysis
    
    # Enhanced volatility analysis
    enhanced_pricing: EnhancedStraddlePricing
    buying_signal: StraddleBuyingSignal
    vol_forecast: VolatilityForecast
    
    # Integrated assessment
    combined_recommendation: str  # "STRONG_BUY", "BUY", "HOLD", "SELL", "AVOID"
    combined_confidence: float
    key_insights: List[str]
    risk_warnings: List[str]
    
    def get_comprehensive_score(self) -> float:
        """Get combined score from MAD and enhanced models (0-100)"""
        # Weight MAD analysis (40%) and enhanced models (60%)
        mad_score = self._convert_mad_to_score()
        enhanced_score = self.buying_signal.signal_strength
        
        return 0.4 * mad_score + 0.6 * enhanced_score
    
    def _convert_mad_to_score(self) -> float:
        """Convert MAD efficiency ratio to 0-100 score"""
        efficiency = self.mad_analysis.efficiency_ratio
        
        if efficiency < 0.8:  # Significantly underpriced
            return 90
        elif efficiency < 0.9:  # Moderately underpriced
            return 70
        elif efficiency < 1.1:  # Fair value
            return 50
        elif efficiency < 1.2:  # Moderately overpriced
            return 30
        else:  # Significantly overpriced
            return 10


class EnhancedMADStraddleAnalyzer(MADStraddleAnalyzer):
    """
    Enhanced version of MADStraddleAnalyzer with integrated volatility modeling
    """
    
    def __init__(self):
        super().__init__()
        
        # Initialize persistent data manager (MAJOR UPGRADE)
        self.data_manager = SupabaseDataManager(cache_size=50000)  # Much larger cache
        self.volatility_data: Optional[VolatilityAnalysisData] = None
        
        # Initialize enhanced volatility components
        self.regime_detector = VolatilityRegimeDetector(lookback_days=252)
        self.forward_vol_estimator = ForwardVolatilityEstimator(self.regime_detector)
        self.volatility_surface = VolatilitySurface(self.btc_price)
        self.enhanced_pricing_model = EnhancedStraddlePricingModel(
            self.regime_detector, self.forward_vol_estimator, self.volatility_surface
        )
        
        # Enhanced data tracking - now backed by database
        self.data_initialization_complete = False
        self.last_comprehensive_update = None
        
        logging.info("Initialized EnhancedMADStraddleAnalyzer with persistent data management")
    
    async def _initialize_comprehensive_data(self):
        """
        Initialize comprehensive historical data for robust volatility analysis
        
        This is the MAJOR UPGRADE that addresses the data limitation issue.
        Instead of ~100 data points, we now load thousands of historical points.
        """
        try:
            if not self.data_manager:
                logging.error("Data manager not initialized")
                return
            
            # Load comprehensive volatility analysis data
            logging.info("Loading comprehensive historical data for volatility analysis...")
            
            # Get extended historical data (7 days = 168 hours for high frequency data)
            # This will provide thousands of data points vs the original ~100
            self.volatility_data = await self.data_manager.get_returns_for_volatility_analysis(
                hours_back=168,  # 1 week of high-frequency data
                min_points=2000  # Require substantial data
            )
            
            if self.volatility_data and len(self.volatility_data.returns) >= 500:
                logging.info(f"✨ Loaded {len(self.volatility_data.returns)} returns for analysis")
                logging.info(f"   Historical period: {len(self.volatility_data.timestamps)} time points")
                logging.info(f"   Realized volatility metrics:")
                logging.info(f"     Short-term: {self.volatility_data.realized_vol_short:.1%}")
                logging.info(f"     Medium-term: {self.volatility_data.realized_vol_medium:.1%}")
                logging.info(f"     Long-term: {self.volatility_data.realized_vol_long:.1%}")
                
                # Initialize regime detector with comprehensive historical data
                await self._initialize_regime_detector_with_history()
                
                # Update forward volatility estimator
                self.forward_vol_estimator.update_historical_data(self.volatility_data.returns)
                
                self.last_comprehensive_update = datetime.now()
                
            else:
                logging.warning(f"Limited historical data available: {len(self.volatility_data.returns) if self.volatility_data else 0} returns")
                logging.warning("Consider running historical_data_collector.py to populate database")
                
        except Exception as e:
            logging.error(f"Error initializing comprehensive data: {e}")
            # Create minimal fallback data to prevent system failure
            self.volatility_data = VolatilityAnalysisData(
                returns=[0.001] * 500,
                prices=[50000.0] * 500,
                timestamps=[datetime.now()] * 500,
                realized_vol_short=0.2,
                realized_vol_medium=0.25,
                realized_vol_long=0.3,
                vol_of_vol=0.1,
                price_vol_correlation=0.0
            )
    
    async def _initialize_regime_detector_with_history(self):
        """Initialize regime detector with comprehensive historical price data"""
        try:
            if not self.volatility_data:
                return
            
            # Feed historical prices to regime detector
            for i, (price, timestamp) in enumerate(zip(self.volatility_data.prices, self.volatility_data.timestamps)):
                # Convert datetime to timestamp
                ts = timestamp.timestamp() if isinstance(timestamp, datetime) else timestamp
                
                # Add to regime detector (sample every 10th point to avoid overwhelming)
                if i % 10 == 0:
                    self.regime_detector.add_price_observation(price, ts)
            
            logging.info(f"Initialized regime detector with {len(self.volatility_data.prices)//10} historical observations")
            
        except Exception as e:
            logging.error(f"Error initializing regime detector with history: {e}")
    
    async def get_btc_price(self):
        """Enhanced BTC price update with persistent data storage"""
        # Call original method
        await super().get_btc_price()
        
        if self.btc_price > 0:
            # Store in persistent database (MAJOR UPGRADE)
            if self.data_manager:
                await self.data_manager.store_real_time_price(self.btc_price)
            
            # Update enhanced volatility models
            current_time = time.time()
            
            # Add to regime detector
            self.regime_detector.add_price_observation(self.btc_price, current_time)
            
            # Update volatility surface spot price
            self.volatility_surface.update_spot_price(self.btc_price)
            
            # Load comprehensive volatility data if not done yet
            if not self.data_initialization_complete:
                await self._initialize_comprehensive_data()
                self.data_initialization_complete = True
    
    async def load_expiration_data(self):
        """Enhanced expiration data loading with volatility surface updates"""
        # Call original method first
        await super().load_expiration_data()
        
        # Update volatility surface with market option data
        current_time = time.time()
        
        for expiry_date, exp_data in self.expirations.items():
            if exp_data.has_straddle():
                # Calculate implied volatilities and add to surface
                call = exp_data.atm_call
                put = exp_data.atm_put
                
                if call and put and call.mark_price > 0 and put.mark_price > 0:
                    # Calculate implied vol for call and put separately
                    time_to_expiry = exp_data.days_to_expiry / 365.0
                    
                    if time_to_expiry > 0:
                        # Call implied vol
                        call_iv = FinancialMath.implied_volatility_newton_raphson(
                            call.mark_price, self.btc_price, call.strike,
                            time_to_expiry, TradingConstants.DEFAULT_RISK_FREE_RATE, "call"
                        )
                        
                        # Put implied vol
                        put_iv = FinancialMath.implied_volatility_newton_raphson(
                            put.mark_price, self.btc_price, put.strike,
                            time_to_expiry, TradingConstants.DEFAULT_RISK_FREE_RATE, "put"
                        )
                        
                        # Add to volatility surface
                        if call_iv and call_iv > 0:
                            self.volatility_surface.add_market_vol_point(
                                call.strike, call.expiry_ts, call_iv, confidence=0.8
                            )
                        
                        if put_iv and put_iv > 0:
                            self.volatility_surface.add_market_vol_point(
                                put.strike, put.expiry_ts, put_iv, confidence=0.8
                            )
    
    def analyze_straddle_with_enhanced_models(self, exp_data: ExpirationData) -> Optional[IntegratedStraddleAnalysis]:
        """
        Perform comprehensive straddle analysis using both MAD and enhanced volatility models
        
        Args:
            exp_data: Expiration data with straddle information
            
        Returns:
            IntegratedStraddleAnalysis or None if insufficient data
        """
        if not exp_data.has_straddle():
            logging.debug(f"No straddle available for {exp_data.expiry_date}")
            return None
        
        # Check for comprehensive volatility data
        if not self.volatility_data or len(self.volatility_data.returns) < 500:
            logging.warning(f"Limited volatility data ({len(self.volatility_data.returns) if self.volatility_data else 0} points) - loading more data")
            await self._initialize_comprehensive_data()
        
        # Get original MAD analysis
        if not exp_data.mad_analysis:
            # Calculate MAD analysis if not already done
            straddle_price = exp_data.get_straddle_price()
            strike_price = exp_data.atm_call.strike
            exp_data.mad_analysis = exp_data.mad_analyzer.analyze_straddle_efficiency(
                straddle_price, self.btc_price, strike_price
            )
            
            if not exp_data.mad_analysis:
                return None
        
        # Get enhanced pricing analysis
        time_to_expiry = exp_data.days_to_expiry / 365.0
        straddle_price = exp_data.get_straddle_price()
        strike_price = exp_data.atm_call.strike
        
        enhanced_pricing = self.enhanced_pricing_model.price_straddle(
            spot_price=self.btc_price,
            strike_price=strike_price,
            time_to_expiry=time_to_expiry,
            market_straddle_price=straddle_price,
            current_returns=self.volatility_data.returns if self.volatility_data else []
        )
        
        if not enhanced_pricing:
            logging.warning(f"Could not generate enhanced pricing for {exp_data.expiry_date}")
            return None
        
        # Generate buying signal
        buying_signal = self.enhanced_pricing_model.generate_straddle_buying_signal(
            enhanced_pricing, straddle_price
        )
        
        # Get volatility forecast using comprehensive historical data
        vol_forecast = self.forward_vol_estimator.forecast_volatility(
            exp_data.days_to_expiry, 
            self.volatility_data.returns if self.volatility_data else []
        )
        
        # Create integrated analysis
        integrated_analysis = self._create_integrated_analysis(
            exp_data, enhanced_pricing, buying_signal, vol_forecast
        )
        
        return integrated_analysis
    
    def _create_integrated_analysis(self, exp_data: ExpirationData, 
                                   enhanced_pricing: EnhancedStraddlePricing,
                                   buying_signal: StraddleBuyingSignal,
                                   vol_forecast: Optional[VolatilityForecast]) -> IntegratedStraddleAnalysis:
        """Create integrated analysis combining all models"""
        
        # Combine recommendations
        mad_recommendation = self._get_mad_recommendation(exp_data.mad_analysis)
        enhanced_recommendation = buying_signal.recommendation
        
        # Determine combined recommendation
        combined_recommendation = self._combine_recommendations(
            mad_recommendation, enhanced_recommendation
        )
        
        # Calculate combined confidence
        mad_confidence = exp_data.mad_analysis.confidence_level if hasattr(exp_data.mad_analysis, 'confidence_level') else 0.75
        enhanced_confidence = buying_signal.confidence_level
        combined_confidence = (mad_confidence + enhanced_confidence) / 2
        
        # Generate key insights
        key_insights = self._generate_key_insights(
            exp_data.mad_analysis, enhanced_pricing, buying_signal, vol_forecast
        )
        
        # Combine risk warnings
        mad_warnings = exp_data.mad_analysis.get_risk_warnings()
        enhanced_warnings = buying_signal.risk_factors
        risk_warnings = list(set(mad_warnings + enhanced_warnings))  # Remove duplicates
        
        return IntegratedStraddleAnalysis(
            expiry_date=exp_data.expiry_date,
            strike_price=exp_data.atm_call.strike,
            time_to_expiry=exp_data.days_to_expiry / 365.0,
            straddle_price=exp_data.get_straddle_price(),
            mad_analysis=exp_data.mad_analysis,
            enhanced_pricing=enhanced_pricing,
            buying_signal=buying_signal,
            vol_forecast=vol_forecast,
            combined_recommendation=combined_recommendation,
            combined_confidence=combined_confidence,
            key_insights=key_insights,
            risk_warnings=risk_warnings
        )
    
    def _get_mad_recommendation(self, mad_analysis: MADAnalysis) -> str:
        """Extract recommendation from MAD analysis"""
        assessment = mad_analysis.get_straddle_assessment()
        
        if "UNDERPRICED" in assessment:
            return "BUY"
        elif "OVERPRICED" in assessment:
            return "SELL"
        else:
            return "HOLD"
    
    def _combine_recommendations(self, mad_rec: str, enhanced_rec: str) -> str:
        """Combine MAD and enhanced model recommendations"""
        
        # Create scoring system
        scores = {"STRONG_BUY": 5, "BUY": 4, "HOLD": 3, "SELL": 2, "AVOID": 1}
        
        mad_score = scores.get(mad_rec, 3)
        enhanced_score = scores.get(enhanced_rec, 3)
        
        # Weight enhanced models more heavily (60%)
        combined_score = 0.4 * mad_score + 0.6 * enhanced_score
        
        # Convert back to recommendation
        if combined_score >= 4.5:
            return "STRONG_BUY"
        elif combined_score >= 3.5:
            return "BUY"
        elif combined_score >= 2.5:
            return "HOLD"
        elif combined_score >= 1.5:
            return "SELL"
        else:
            return "AVOID"
    
    def _generate_key_insights(self, mad_analysis: MADAnalysis, 
                              enhanced_pricing: EnhancedStraddlePricing,
                              buying_signal: StraddleBuyingSignal,
                              vol_forecast: Optional[VolatilityForecast]) -> List[str]:
        """Generate key insights from integrated analysis"""
        
        insights = []
        
        # MAD insights
        mad_ratio = mad_analysis.mad_sd_ratio
        if mad_ratio < 0.65:
            insights.append(f"Heavy tail risk detected (MAD/SD: {mad_ratio:.2f}) - increases option value")
        elif mad_ratio > 0.85:
            insights.append(f"Light tails detected (MAD/SD: {mad_ratio:.2f}) - reduces option value")
        
        # Volatility insights
        if enhanced_pricing.expected_vol_change > 0.15:
            insights.append(f"Strong vol expansion expected (+{enhanced_pricing.expected_vol_change:.1%}) - favorable for buying")
        elif enhanced_pricing.expected_vol_change < -0.15:
            insights.append(f"Vol contraction expected ({enhanced_pricing.expected_vol_change:+.1%}) - unfavorable for buying")
        
        # Regime insights
        regime_evidence = buying_signal.regime_evidence
        if "Low vol regime" in regime_evidence:
            insights.append("Currently in low vol regime - vol expansion more likely")
        elif "High vol regime" in regime_evidence or "Extreme vol regime" in regime_evidence:
            insights.append("Currently in high vol regime - mean reversion risk")
        
        # Pricing insights
        price_vs_fair = buying_signal.price_vs_enhanced_fair_value
        if price_vs_fair < -0.10:
            insights.append(f"Straddle underpriced by {abs(price_vs_fair):.1%} vs enhanced fair value")
        elif price_vs_fair > 0.10:
            insights.append(f"Straddle overpriced by {price_vs_fair:.1%} vs enhanced fair value")
        
        # Model agreement insights
        bs_price = enhanced_pricing.bs_straddle_price
        enhanced_price = enhanced_pricing.composite_straddle_price
        model_diff = abs(enhanced_price - bs_price) / bs_price
        
        if model_diff > 0.15:
            insights.append(f"Enhanced models differ significantly from Black-Scholes ({model_diff:.1%})")
        
        return insights
    
    def show_enhanced_expiration_analysis(self):
        """Show enhanced expiration analysis with integrated models"""
        print("\n" + "="*120)
        print("ENHANCED STRADDLE ANALYSIS - INTEGRATED MAD + VOLATILITY MODELS")
        
        # Show regime status
        regime_metrics = self.regime_detector.analyze_regime()
        if regime_metrics:
            print(f"CURRENT VOLATILITY REGIME: {regime_metrics.regime.value.upper()} "
                  f"(Confidence: {regime_metrics.confidence:.1%})")
        
        print("="*120)
        print(f"{'#':<3} {'Date':<12} {'Days':<6} {'Strike':<8} {'Price':<8} {'MAD':<8} {'Enhanced':<10} {'Combined':<10} {'Recommendation'}")
        print("-"*120)
        
        analyses = []
        for i, (date, exp_data) in enumerate(sorted(self.expirations.items()), 1):
            if exp_data.has_straddle():
                # Get integrated analysis
                integrated = self.analyze_straddle_with_enhanced_models(exp_data)
                
                if integrated:
                    analyses.append((i, date, exp_data, integrated))
                    
                    # Display row
                    strike = f"${integrated.strike_price:.0f}"
                    price = f"${integrated.straddle_price:.2f}"
                    mad_ratio = f"{integrated.mad_analysis.efficiency_ratio:.2f}"
                    enhanced_score = f"{integrated.buying_signal.signal_strength:.0f}"
                    combined_score = f"{integrated.get_comprehensive_score():.0f}"
                    recommendation = integrated.combined_recommendation
                    
                    # Color coding for terminal display
                    if recommendation in ["STRONG_BUY", "BUY"]:
                        rec_display = f"🟢 {recommendation}"
                    elif recommendation == "HOLD":
                        rec_display = f"🟡 {recommendation}"
                    else:
                        rec_display = f"🔴 {recommendation}"
                    
                    print(f"{i:<3} {date:<12} {exp_data.days_to_expiry:<6.1f} {strike:<8} {price:<8} "
                          f"{mad_ratio:<8} {enhanced_score:<10} {combined_score:<10} {rec_display}")
        
        return analyses
    
    def plot_integrated_straddle_analysis(self, integrated_analysis: IntegratedStraddleAnalysis):
        """Plot comprehensive integrated analysis"""
        
        fig = plt.figure(figsize=(16, 12))
        
        # Create grid layout
        gs = fig.add_gridspec(3, 2, height_ratios=[2, 1, 1], hspace=0.3, wspace=0.3)
        
        # Main price chart with breakevens
        ax_main = fig.add_subplot(gs[0, :])
        self._plot_main_straddle_chart(ax_main, integrated_analysis)
        
        # Volatility analysis
        ax_vol = fig.add_subplot(gs[1, 0])
        self._plot_volatility_analysis(ax_vol, integrated_analysis)
        
        # Risk metrics
        ax_risk = fig.add_subplot(gs[1, 1])
        self._plot_risk_metrics(ax_risk, integrated_analysis)
        
        # Insights and recommendations
        ax_insights = fig.add_subplot(gs[2, :])
        self._plot_insights_panel(ax_insights, integrated_analysis)
        
        plt.suptitle(f"Integrated Straddle Analysis - {integrated_analysis.expiry_date}", 
                    fontsize=16, fontweight='bold')
        plt.show()
    
    def _plot_main_straddle_chart(self, ax, analysis: IntegratedStraddleAnalysis):
        """Plot main straddle price chart"""
        
        strike = analysis.strike_price
        straddle_price = analysis.straddle_price
        
        # Breakeven lines
        lower_breakeven = strike - straddle_price
        upper_breakeven = strike + straddle_price
        
        # Plot current BTC price
        ax.axhline(y=self.btc_price, color='blue', linewidth=3, 
                  label=f'BTC Price: ${self.btc_price:.0f}')
        
        # Plot breakeven lines
        ax.axhline(y=upper_breakeven, color='red', linestyle='--', 
                  linewidth=2, label=f'Upper Breakeven: ${upper_breakeven:.0f}')
        ax.axhline(y=lower_breakeven, color='red', linestyle='--', 
                  linewidth=2, label=f'Lower Breakeven: ${lower_breakeven:.0f}')
        
        # Plot strike
        ax.axhline(y=strike, color='green', linestyle=':', 
                  linewidth=1, alpha=0.7, label=f'Strike: ${strike:.0f}')
        
        # Set limits and labels
        padding = (upper_breakeven - lower_breakeven) * 0.15
        ax.set_ylim(lower_breakeven - padding, upper_breakeven + padding)
        ax.set_ylabel('Price ($)')
        ax.set_title('Straddle Breakeven Analysis')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        # Add info box
        info_text = self._create_main_chart_info_text(analysis)
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
               verticalalignment='top', 
               bbox=dict(boxstyle="round", facecolor='lightblue', alpha=0.8),
               fontsize=10)
    
    def _plot_volatility_analysis(self, ax, analysis: IntegratedStraddleAnalysis):
        """Plot volatility analysis"""
        
        enhanced = analysis.enhanced_pricing
        
        # Volatility comparison
        vols = [
            enhanced.bs_implied_vol,
            enhanced.forward_vol_expected,
            enhanced.regime_adjusted_vol,
            enhanced.composite_expected_vol
        ]
        
        labels = ['Current IV', 'Forward Vol', 'Regime Adj', 'Composite']
        colors = ['blue', 'green', 'orange', 'red']
        
        bars = ax.bar(labels, [v * 100 for v in vols], color=colors, alpha=0.7)
        
        # Add value labels on bars
        for bar, vol in zip(bars, vols):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                   f'{vol:.1%}', ha='center', va='bottom', fontsize=9)
        
        ax.set_ylabel('Volatility (%)')
        ax.set_title('Volatility Analysis')
        ax.grid(True, alpha=0.3)
        
        # Add expected change annotation
        vol_change = enhanced.expected_vol_change
        change_text = f"Expected Vol Change: {vol_change:+.1%}"
        ax.text(0.5, 0.95, change_text, transform=ax.transAxes, 
               ha='center', va='top', fontweight='bold',
               bbox=dict(boxstyle="round", facecolor='yellow', alpha=0.7))
    
    def _plot_risk_metrics(self, ax, analysis: IntegratedStraddleAnalysis):
        """Plot risk metrics"""
        
        # Risk indicators
        mad_ratio = analysis.mad_analysis.mad_sd_ratio
        vol_expansion_prob = analysis.enhanced_pricing.vol_expansion_probability
        pricing_confidence = analysis.combined_confidence
        
        metrics = ['Tail Risk\n(MAD/SD)', 'Vol Expansion\nProbability', 'Pricing\nConfidence']
        values = [mad_ratio, vol_expansion_prob, pricing_confidence]
        
        # Color coding based on values
        colors = []
        for i, val in enumerate(values):
            if i == 0:  # MAD ratio - lower is higher risk
                colors.append('red' if val < 0.65 else 'orange' if val < 0.75 else 'green')
            else:  # Higher is better
                colors.append('green' if val > 0.8 else 'orange' if val > 0.6 else 'red')
        
        bars = ax.bar(metrics, values, color=colors, alpha=0.7)
        
        # Add value labels
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                   f'{val:.2f}', ha='center', va='bottom', fontsize=9)
        
        ax.set_ylim(0, 1.1)
        ax.set_ylabel('Score')
        ax.set_title('Risk Metrics')
        ax.grid(True, alpha=0.3)
    
    def _plot_insights_panel(self, ax, analysis: IntegratedStraddleAnalysis):
        """Plot insights and recommendations panel"""
        ax.axis('off')
        
        # Recommendation box
        rec = analysis.combined_recommendation
        score = analysis.get_comprehensive_score()
        
        rec_color = 'green' if rec in ['STRONG_BUY', 'BUY'] else 'orange' if rec == 'HOLD' else 'red'
        
        rec_text = f"RECOMMENDATION: {rec}\nCombined Score: {score:.0f}/100\nConfidence: {analysis.combined_confidence:.1%}"
        
        ax.text(0.02, 0.95, rec_text, transform=ax.transAxes, 
               verticalalignment='top', fontsize=12, fontweight='bold',
               bbox=dict(boxstyle="round", facecolor=rec_color, alpha=0.3))
        
        # Key insights
        insights_text = "KEY INSIGHTS:\n" + "\n".join([f"• {insight}" for insight in analysis.key_insights[:4]])
        
        ax.text(0.35, 0.95, insights_text, transform=ax.transAxes,
               verticalalignment='top', fontsize=10,
               bbox=dict(boxstyle="round", facecolor='lightblue', alpha=0.3))
        
        # Risk warnings
        if analysis.risk_warnings:
            warnings_text = "RISK WARNINGS:\n" + "\n".join([f"⚠ {warning}" for warning in analysis.risk_warnings[:3]])
            
            ax.text(0.02, 0.45, warnings_text, transform=ax.transAxes,
                   verticalalignment='top', fontsize=10,
                   bbox=dict(boxstyle="round", facecolor='yellow', alpha=0.3))
    
    def _create_main_chart_info_text(self, analysis: IntegratedStraddleAnalysis) -> str:
        """Create info text for main chart"""
        
        lines = [
            f"Straddle Price: ${analysis.straddle_price:.2f}",
            f"Days to Expiry: {analysis.time_to_expiry * 365:.0f}",
            f"Range Width: ${2 * analysis.straddle_price:.0f}",
            "",
            "PRICING MODELS:",
            f"Black-Scholes: ${analysis.enhanced_pricing.bs_straddle_price:.2f}",
            f"Enhanced Fair: ${analysis.enhanced_pricing.composite_straddle_price:.2f}",
            f"MAD Efficiency: {analysis.mad_analysis.efficiency_ratio:.2f}",
            "",
            f"Vol Regime: {analysis.buying_signal.current_vol_regime.value.title()}"
        ]
        
        return "\n".join(lines)
    
    async def run_enhanced_interactive_session(self):
        """Run enhanced interactive session with integrated analysis"""
        print("Starting Enhanced MAD + Volatility Straddle Analyzer...")
        print("Collecting price data for comprehensive volatility analysis...")
        
        # Connect and get initial data
        if not await self.connect_and_authenticate():
            print("Failed to connect")
            return
        
        # Initialize comprehensive historical data (MAJOR UPGRADE)
        print("Initializing comprehensive historical data...")
        if not await self.data_manager.initialize():
            print("❌ Failed to initialize data manager")
            return
        
        # Load substantial historical data
        await self._initialize_comprehensive_data()
        
        # Collect recent real-time data for immediate context
        print("Collecting recent price updates...")
        for i in range(20):  # Fewer iterations since we have historical data
            await self.get_btc_price()
            if i < 19:
                await asyncio.sleep(1.0)  # More reasonable interval
        
        await self.load_expiration_data()
        
        # Interactive loop
        while True:
            analyses = self.show_enhanced_expiration_analysis()
            
            if not analyses:
                print("\nNo valid expirations found with complete data")
                break
            
            print("\nOptions:")
            print("  Enter number (1-{}) to view detailed integrated analysis".format(len(analyses)))
            print("  'position' to calculate position sizing")
            print("  'regime' to view volatility regime analysis") 
            print("  'surface' to view volatility surface status")
            print("  'refresh' to reload data")
            print("  'quit' to exit")
            
            choice = input("\nYour choice: ").strip()
            
            if choice.lower() == 'quit':
                break
            elif choice.lower() == 'regime':
                self._display_regime_analysis()
                continue
            elif choice.lower() == 'surface':
                self._display_surface_analysis()
                continue
            elif choice.lower() == 'position':
                print("\nStarting Enhanced Position Calculator...")
                try:
                    position_selector = EnhancedStraddlePositionSelector(self)
                    position_selector.interactive_position_calculator()
                except Exception as e:
                    print(f"Error in position calculator: {e}")
                continue
            elif choice.lower() == 'refresh':
                print("\nRefreshing data and updating all models...")
                
                if hasattr(self.thalex, 'disconnect'):
                    try:
                        await self.thalex.disconnect()
                    except:
                        pass
                
                if await self.connect_and_authenticate():
                    for i in range(10):
                        await self.get_btc_price()
                        if i < 9:
                            await asyncio.sleep(1)
                    await self.load_expiration_data()
                    print("All models refreshed successfully!")
                else:
                    print("Failed to refresh - connection error")
                continue
            else:
                try:
                    choice_num = int(choice)
                    if 1 <= choice_num <= len(analyses):
                        _, date, exp_data, integrated_analysis = analyses[choice_num - 1]
                        print(f"\nShowing detailed integrated analysis for {date}...")
                        self.plot_integrated_straddle_analysis(integrated_analysis)
                    else:
                        print("Invalid choice. Please try again.")
                except ValueError:
                    print("Invalid input. Please enter a number.")
        
        # Cleanup
        if hasattr(self.thalex, 'disconnect'):
            await self.thalex.disconnect()
        
        # Clean up data manager
        if self.data_manager:
            await self.data_manager.cleanup()
    
    def _display_regime_analysis(self):
        """Display detailed volatility regime analysis"""
        regime_metrics = self.regime_detector.analyze_regime()
        
        if not regime_metrics:
            print("Insufficient data for regime analysis")
            return
        
        print("\n" + "="*70)
        print("VOLATILITY REGIME ANALYSIS")
        print("="*70)
        
        regime_metrics_dict = {
            'Current Regime': regime_metrics.regime.value.upper(),
            'Confidence': f"{regime_metrics.confidence:.1%}",
            'Current Vol': f"{regime_metrics.current_vol:.1%}",
            'Short-term Vol': f"{regime_metrics.short_term_vol:.1%}",
            'Medium-term Vol': f"{regime_metrics.medium_term_vol:.1%}",
            'Long-term Vol': f"{regime_metrics.long_term_vol:.1%}",
            'Vol Momentum': f"{regime_metrics.vol_momentum:+.2f}",
            'Mean Reversion Distance': f"{regime_metrics.vol_mean_reversion:+.1f} σ",
            'Vol Percentile': f"{regime_metrics.vol_percentile:.0f}th",
            'Vol-Price Correlation': f"{regime_metrics.vol_price_correlation:+.2f}",
            'Time Since Regime Change': f"{regime_metrics.time_since_regime_change:.1f} hours"
        }
        
        for key, value in regime_metrics_dict.items():
            print(f"{key:<25}: {value}")
        
        # Straddle implications
        implications = self.regime_detector.get_regime_implications_for_straddles()
        print(f"\nStraddle Trading Implication:")
        print(f"  {implications.get('implication', 'Unknown')}")
        
        print("="*70)
    
    def _display_surface_analysis(self):
        """Display volatility surface analysis"""
        surface_metrics = self.volatility_surface.calculate_surface_quality_metrics()
        
        print("\n" + "="*70)
        print("VOLATILITY SURFACE STATUS")
        print("="*70)
        
        print(f"Surface Quality Score: {surface_metrics['surface_quality_score']:.1%}")
        print(f"Coverage Score: {surface_metrics['coverage_score']:.1%}")
        print(f"Data Staleness Score: {surface_metrics['staleness_score']:.1%}")
        print(f"Total Data Points: {surface_metrics['data_points']}")
        print(f"Expiry Slices: {surface_metrics['expiry_slices']}")
        print(f"Total Strikes: {surface_metrics['total_strikes']}")
        
        print("="*70)


class EnhancedStraddlePositionSelector(InteractiveStraddleSelector):
    """Enhanced position selector with integrated volatility models"""
    
    def __init__(self, enhanced_analyzer: EnhancedMADStraddleAnalyzer):
        self.enhanced_analyzer = enhanced_analyzer
        super().__init__(enhanced_analyzer)  # Initialize parent with analyzer
    
    def show_enhanced_selling_opportunities(self):
        """Show selling opportunities with enhanced analysis"""
        print("\n" + "="*120)
        print("🔴 ENHANCED STRADDLE SELLING OPPORTUNITIES")
        print("="*120)
        
        opportunities = []
        for i, (date, exp_data) in enumerate(sorted(self.enhanced_analyzer.expirations.items()), 1):
            if exp_data.has_straddle():
                integrated = self.enhanced_analyzer.analyze_straddle_with_enhanced_models(exp_data)
                
                if integrated and integrated.combined_recommendation in ["SELL", "AVOID"]:
                    opportunities.append((i, date, exp_data, integrated))
        
        if not opportunities:
            print("No overpriced straddles found based on enhanced analysis")
            return []
        
        print(f"{'#':<3} {'Date':<12} {'Days':<6} {'Strike':<8} {'Price':<8} {'Score':<6} {'Recommendation':<15} {'Key Insight'}")
        print("-"*120)
        
        for i, date, exp_data, integrated in opportunities:
            score = integrated.get_comprehensive_score()
            key_insight = integrated.key_insights[0] if integrated.key_insights else "Standard analysis"
            
            print(f"{i:<3} {date:<12} {exp_data.days_to_expiry:<6.1f} ${integrated.strike_price:<7.0f} "
                  f"${integrated.straddle_price:<7.2f} {score:<6.0f} {integrated.combined_recommendation:<15} {key_insight[:50]}")
        
        return opportunities


async def main():
    """Main entry point for enhanced MAD straddle analyzer"""
    import logging
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    print("Starting Enhanced MAD Straddle Analyzer...")
    
    analyzer = EnhancedMADStraddleAnalyzer()
    
    try:
        await analyzer.run_enhanced_interactive_session()
    except KeyboardInterrupt:
        print("\nAnalyzer stopped by user")
    except Exception as e:
        print(f"Error: {e}")
        logging.error(f"Analyzer error: {e}", exc_info=True)
    finally:
        print("Enhanced MAD Straddle Analyzer session ended")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())

