#!/usr/bin/env python3
"""
Straddle Position Calculator - Margin and position sizing calculator for selling straddles
Calculates required capital, margin requirements, and risk metrics for safe straddle selling
"""
import math
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

@dataclass
class StraddlePosition:
    """Represents a straddle position to be sold"""
    expiry_date: str
    days_to_expiry: float
    strike: float
    call_price: float
    put_price: float
    straddle_price: float
    efficiency_ratio: float
    mad_sd_ratio: float
    confidence: float
    breakeven_lower: float
    breakeven_upper: float
    
    def get_premium_collected(self, quantity: float) -> float:
        """Calculate total premium collected for given quantity"""
        return self.straddle_price * quantity
    
    def get_breakeven_range_width(self) -> float:
        """Get width of breakeven range"""
        return self.breakeven_upper - self.breakeven_lower
    
    def get_range_percentage(self, spot_price: float) -> float:
        """Get breakeven range as percentage of spot price"""
        return (self.get_breakeven_range_width() / spot_price) * 100

@dataclass
class MarginRequirement:
    """Margin requirement calculation results"""
    initial_margin: float
    maintenance_margin: float
    total_premium_collected: float
    net_capital_required: float
    recommended_capital: float
    max_loss_estimate: float
    safety_buffer: float

@dataclass
class RiskMetrics:
    """Risk assessment metrics for position"""
    max_profit: float
    max_loss_potential: str  # "Unlimited" or specific amount
    probability_of_profit: float
    kelly_optimal_size: float
    risk_reward_ratio: float
    position_risk_percentage: float
    concentration_warning: bool

class StraddlePositionCalculator:
    """Calculator for straddle position sizing and margin requirements"""
    
    def __init__(self, spot_price: float):
        self.spot_price = spot_price
        self.selected_straddles: List[StraddlePosition] = []
        
        # Thalex-specific margin parameters (conservative estimates)
        self.margin_multiplier = 0.15  # 15% of notional for options
        self.maintenance_ratio = 0.75  # 75% of initial margin
        self.safety_buffer_ratio = 1.5  # 150% safety buffer
        
        # Risk management parameters
        self.max_position_concentration = 0.25  # Max 25% of capital per position
        self.max_portfolio_risk = 0.50  # Max 50% of capital at risk
        
    def add_straddle_from_expiry_data(self, expiry_data, quantity: float = 1.0):
        """Add straddle position from ExpirationData object"""
        if not expiry_data.has_straddle() or not expiry_data.mad_analysis:
            raise ValueError(f"Incomplete data for {expiry_data.expiry_date}")
            
        # Calculate breakeven points
        straddle_price = expiry_data.get_straddle_price()
        strike = expiry_data.atm_call.strike
        
        position = StraddlePosition(
            expiry_date=expiry_data.expiry_date,
            days_to_expiry=expiry_data.days_to_expiry,
            strike=strike,
            call_price=expiry_data.atm_call.mark_price,
            put_price=expiry_data.atm_put.mark_price,
            straddle_price=straddle_price,
            efficiency_ratio=expiry_data.mad_analysis.efficiency_ratio,
            mad_sd_ratio=expiry_data.mad_analysis.mad_sd_ratio,
            confidence=0.85,  # Default confidence
            breakeven_lower=strike - straddle_price,
            breakeven_upper=strike + straddle_price
        )
        
        self.selected_straddles.append(position)
        return position
    
    def calculate_single_position_margin(self, position: StraddlePosition, quantity: float) -> MarginRequirement:
        """Calculate margin requirements for a single straddle position"""
        
        # Premium collected (credit to account)
        premium_collected = position.get_premium_collected(quantity)
        
        # Notional value for margin calculation
        notional_value = position.strike * quantity
        
        # Initial margin (conservative approach)
        # For short straddles: higher of percentage of notional or multiple of premium
        margin_method_1 = notional_value * self.margin_multiplier
        margin_method_2 = premium_collected * 2.0  # 200% of premium collected
        initial_margin = max(margin_method_1, margin_method_2)
        
        # Maintenance margin
        maintenance_margin = initial_margin * self.maintenance_ratio
        
        # Net capital required (margin minus premium collected)
        net_capital_required = max(0, initial_margin - premium_collected)
        
        # Safety buffer for unexpected moves
        safety_buffer = net_capital_required * (self.safety_buffer_ratio - 1)
        
        # Recommended capital (includes safety buffer)
        recommended_capital = net_capital_required + safety_buffer
        
        # Maximum loss estimate (simplified)
        max_loss_estimate = self._estimate_max_loss(position, quantity)
        
        return MarginRequirement(
            initial_margin=initial_margin,
            maintenance_margin=maintenance_margin,
            total_premium_collected=premium_collected,
            net_capital_required=net_capital_required,
            recommended_capital=recommended_capital,
            max_loss_estimate=max_loss_estimate,
            safety_buffer=safety_buffer
        )
    
    def _estimate_max_loss(self, position: StraddlePosition, quantity: float) -> float:
        """Estimate maximum loss for stress testing (simplified)"""
        # Assume BTC could move 2x the breakeven range in extreme scenarios
        extreme_move_multiplier = 2.0
        breakeven_width = position.get_breakeven_range_width()
        extreme_move = breakeven_width * extreme_move_multiplier
        
        # Maximum loss occurs at extreme price levels
        # Loss = (Extreme move - Breakeven width) * Quantity - Premium collected
        premium_collected = position.get_premium_collected(quantity)
        max_loss = (extreme_move - breakeven_width) * quantity - premium_collected
        
        return max(max_loss, 0)  # Cannot be negative (limited by premium collected)
    
    def calculate_risk_metrics(self, position: StraddlePosition, quantity: float, account_size: float) -> RiskMetrics:
        """Calculate comprehensive risk metrics for position"""
        
        # Maximum profit (premium collected)
        max_profit = position.get_premium_collected(quantity)
        
        # Probability of profit (simplified based on efficiency ratio)
        # Higher efficiency ratio = higher theoretical edge = higher probability
        base_prob = 0.50  # Base 50% probability
        efficiency_edge = max(0, position.efficiency_ratio - 1.0)  # Edge over fair value
        prob_of_profit = min(0.85, base_prob + (efficiency_edge * 0.10))  # Cap at 85%
        
        # Kelly optimal position size
        kelly_size = self._calculate_kelly_size(position, account_size)
        
        # Risk-reward ratio (simplified)
        breakeven_range_pct = position.get_range_percentage(self.spot_price)
        risk_reward_ratio = max_profit / (max_profit * 2.0)  # Conservative estimate
        
        # Position risk as percentage of account
        margin_req = self.calculate_single_position_margin(position, quantity)
        position_risk_pct = (margin_req.recommended_capital / account_size) * 100
        
        # Concentration warning
        concentration_warning = position_risk_pct > (self.max_position_concentration * 100)
        
        return RiskMetrics(
            max_profit=max_profit,
            max_loss_potential="Unlimited (but capped by margin)",
            probability_of_profit=prob_of_profit,
            kelly_optimal_size=kelly_size,
            risk_reward_ratio=risk_reward_ratio,
            position_risk_percentage=position_risk_pct,
            concentration_warning=concentration_warning
        )
    
    def _calculate_kelly_size(self, position: StraddlePosition, account_size: float) -> float:
        """Calculate Kelly Criterion optimal position size"""
        # Kelly formula: f = (bp - q) / b
        # where: b = odds, p = probability of win, q = probability of loss
        
        # Simplified Kelly for straddle selling
        efficiency_edge = position.efficiency_ratio - 1.0  # Edge over fair value
        win_probability = 0.50 + (efficiency_edge * 0.10)  # Rough estimate
        
        if efficiency_edge <= 0 or win_probability <= 0.5:
            return 0.0  # No edge, no position
        
        # Kelly percentage (very conservative for options)
        kelly_pct = min(0.10, efficiency_edge * 0.05)  # Cap at 10% of account
        
        return account_size * kelly_pct
    
    def generate_position_report(self, position: StraddlePosition, quantity: float, account_size: float) -> str:
        """Generate detailed position report"""
        margin_req = self.calculate_single_position_margin(position, quantity)
        risk_metrics = self.calculate_risk_metrics(position, quantity, account_size)
        
        report_lines = [
            f"",
            f"{'='*80}",
            f"STRADDLE POSITION ANALYSIS - {position.expiry_date}",
            f"{'='*80}",
            f"",
            f"📊 POSITION DETAILS:",
            f"   Strike Price:        ${position.strike:,.0f}",    
            f"   Straddle Price:      ${position.straddle_price:.2f}",
            f"   Quantity to Sell:    {quantity:g} straddles",
            f"   Days to Expiry:      {position.days_to_expiry:.1f}",
            f"   Efficiency Ratio:    {position.efficiency_ratio:.2f} ({'OVERPRICED' if position.efficiency_ratio > 1.15 else 'FAIR'})",
            f"",
            f"💰 FINANCIAL IMPACT:",
            f"   Premium Collected:   ${margin_req.total_premium_collected:,.2f}",
            f"   Initial Margin:      ${margin_req.initial_margin:,.2f}",
            f"   Net Capital Required: ${margin_req.net_capital_required:,.2f}",
            f"   Recommended Capital: ${margin_req.recommended_capital:,.2f} (with safety buffer)",
            f"   Safety Buffer:       ${margin_req.safety_buffer:,.2f}",
            f"",
            f"📈 BREAKEVEN ANALYSIS:",
            f"   Lower Breakeven:     ${position.breakeven_lower:,.0f}",
            f"   Upper Breakeven:     ${position.breakeven_upper:,.0f}",
            f"   Breakeven Range:     ${position.get_breakeven_range_width():,.0f} ({position.get_range_percentage(self.spot_price):.1f}% of spot)",
            f"   Current BTC Price:   ${self.spot_price:,.0f}",
            f"",
            f"⚠️  RISK METRICS:",
            f"   Maximum Profit:      ${risk_metrics.max_profit:,.2f} ({(risk_metrics.max_profit/account_size)*100:.1f}% of account)",
            f"   Maximum Loss:        {risk_metrics.max_loss_potential}",
            f"   Probability of Profit: {risk_metrics.probability_of_profit:.1%}",
            f"   Position Risk:       {risk_metrics.position_risk_percentage:.1f}% of account",
            f"   Kelly Optimal Size:  ${risk_metrics.kelly_optimal_size:,.0f}",
            f"",
        ]
        
        # Warnings
        warnings = []
        if risk_metrics.concentration_warning:
            warnings.append("⚠️  WARNING: Position exceeds recommended concentration limit (25%)")
        if position.mad_sd_ratio < 0.70:
            warnings.append("⚠️  WARNING: Heavy tail risk detected (MAD/SD < 0.70)")
        if position.days_to_expiry < 3:
            warnings.append("⚠️  WARNING: Very short time to expiry - high gamma risk")
        if position.confidence < 0.75:
            warnings.append("⚠️  WARNING: Low confidence in MAD analysis")
            
        if warnings:
            report_lines.extend([
                f"🚨 RISK WARNINGS:",
                *[f"   {warning}" for warning in warnings],
                f""
            ])
        
        # Recommendations
        report_lines.extend([
            f"💡 RECOMMENDATIONS:",
            f"   • Ensure account has at least ${margin_req.recommended_capital:,.0f} available",
            f"   • Monitor position if BTC moves beyond ${position.breakeven_lower:,.0f} - ${position.breakeven_upper:,.0f}",
            f"   • Consider profit taking at 25-50% of maximum profit",
            f"   • Set stop loss if losses exceed 2x premium collected",
            f"",
            f"{'='*80}"
        ])
        
        return "\n".join(report_lines)

class InteractiveStraddleSelector:
    """Interactive interface for selecting and sizing straddle positions"""
    
    def __init__(self, mad_analyzer):
        self.mad_analyzer = mad_analyzer
        self.calculator = StraddlePositionCalculator(mad_analyzer.btc_price)
        
    def get_selling_opportunities(self, min_efficiency: float = 1.15) -> List[Tuple[int, str, any]]:
        """Get list of straddles that show selling opportunities"""
        opportunities = []
        
        for i, (date, exp_data) in enumerate(sorted(self.mad_analyzer.expirations.items()), 1):
            if (exp_data.has_straddle() and 
                exp_data.mad_analysis and 
                exp_data.mad_analysis.efficiency_ratio >= min_efficiency):
                opportunities.append((i, date, exp_data))
                
        return opportunities
    
    def show_selling_opportunities(self):
        """Display available selling opportunities"""
        opportunities = self.get_selling_opportunities()
        
        if not opportunities:
            print("\n❌ No overpriced straddles found (efficiency >= 1.15)")
            return []
            
        print("\n" + "="*100)
        print("🔴 STRADDLE SELLING OPPORTUNITIES (OVERPRICED)")
        print("="*100)
        print(f"{'#':<3} {'Date':<12} {'Days':<6} {'Strike':<8} {'Premium':<10} {'Efficiency':<10} {'Range %':<8} {'Status'}")
        print("-"*100)
        
        for i, date, exp_data in opportunities:
            strike = f"${exp_data.atm_call.strike:.0f}"
            premium = f"${exp_data.get_straddle_price():.2f}"
            efficiency = f"{exp_data.mad_analysis.efficiency_ratio:.2f}"
            
            # Calculate range percentage
            breakeven_width = 2 * exp_data.get_straddle_price()  # Approximate
            range_pct = f"{(breakeven_width / self.calculator.spot_price) * 100:.1f}%"
            
            status = "🔴 SELL" if exp_data.mad_analysis.efficiency_ratio > 1.15 else "⚪ WATCH"
            
            print(f"{i:<3} {date:<12} {exp_data.days_to_expiry:<6.1f} {strike:<8} {premium:<10} {efficiency:<10} {range_pct:<8} {status}")
            
        return opportunities
    
    def interactive_position_calculator(self):
        """Run interactive position calculator"""
        while True:
            print("\n" + "="*60)
            print("📊 STRADDLE POSITION CALCULATOR")
            print("="*60)
            
            # Show opportunities
            opportunities = self.show_selling_opportunities()
            if not opportunities:
                break
                
            print("\nOptions:")
            print("  Enter number (1-{}) to calculate position for that straddle".format(len(opportunities)))
            print("  'quit' to exit")
            
            choice = input("\nSelect straddle to analyze: ").strip()
            
            if choice.lower() == 'quit':
                break
                
            try:
                choice_num = int(choice)
                if 1 <= choice_num <= len(opportunities):
                    _, date, exp_data = opportunities[choice_num - 1]
                    self._calculate_position_for_straddle(exp_data)
                else:
                    print("❌ Invalid selection. Please try again.")
            except ValueError:
                print("❌ Invalid input. Please enter a number.")
    
    def _calculate_position_for_straddle(self, exp_data):
        """Calculate position sizing for selected straddle"""
        print(f"\n📊 Position Calculator for {exp_data.expiry_date}")
        print(f"Strike: ${exp_data.atm_call.strike:.0f}, Premium: ${exp_data.get_straddle_price():.2f}")
        
        # Get position size (now accepts decimals)
        try:
            quantity = float(input("\nHow many straddles do you want to sell (e.g., 1, 0.5, 0.1)? "))
            if quantity <= 0:
                print("❌ Quantity must be positive")
                return
        except ValueError:
            print("❌ Invalid quantity. Please enter a number (decimals allowed)")
            return
            
        # Get account size
        try:
            account_size = float(input("What is your account size (USD)? $"))
            if account_size <= 0:
                print("❌ Account size must be positive")
                return
        except ValueError:
            print("❌ Invalid account size")
            return
        
        # Add position and calculate
        try:
            position = self.calculator.add_straddle_from_expiry_data(exp_data, quantity)
            report = self.calculator.generate_position_report(position, quantity, account_size)
            print(report)
            
            # Ask if user wants to proceed
            proceed = input("\nWould you like to see another position calculation? (y/n): ").strip().lower()
            if proceed != 'y':
                return
                
        except Exception as e:
            print(f"❌ Error calculating position: {e}")

def main():
    """Demo function - would normally integrate with MAD analyzer"""
    print("Straddle Position Calculator")
    print("This tool calculates margin requirements and position sizing for selling straddles")
    print("\nTo use this tool:")
    print("1. First run MADStraddle.py to identify overpriced straddles")
    print("2. Import this module in your trading script")
    print("3. Use InteractiveStraddleSelector with your MAD analyzer")

if __name__ == "__main__":
    main()