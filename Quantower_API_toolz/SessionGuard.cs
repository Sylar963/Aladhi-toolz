// Copyright QUANTOWER LLC. © 2017-2024. All rights reserved.

using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Drawing.Text;
using System.Linq;
using TradingPlatform.BusinessLayer;

namespace TradingSessions
{
    /// <summary>
    /// Session Angel - Trading Guardian & Risk Management Indicator
    /// Monitors trading activity and protects against excessive losses
    /// </summary>
    public class SessionAngel : Indicator
    {
        #region Private Fields
        
        // Session tracking
        private DateTime _lastProcessedDate = DateTime.MinValue;
        private DateTime _sessionStartTime = DateTime.MinValue;
        private int _animationFrame = 0;
        private DateTime _lastAnimationUpdate = DateTime.MinValue;
        
        // Trading activity tracking
        private double _sessionStartBalance = 0.0;
        private double _currentSessionPnL = 0.0;
        private double _maxDrawdown = 0.0;
        private double _maxProfit = 0.0;
        private double _lastDayProfit = 0.0;
        
        // Risk management
        private bool _warningActive = false;
        private bool _dangerActive = false;
        private DateTime _lastWarningTime = DateTime.MinValue;
        private List<DateTime> _warningHistory = new List<DateTime>();
        
        // Session statistics
        private double _mfe = 0.0; // Maximum Favorable Excursion
        private double _mae = 0.0; // Maximum Adverse Excursion
        private int _totalTrades = 0;
        private int _winningTrades = 0;
        private int _losingTrades = 0;
        private double _largestWin = 0.0;
        private double _largestLoss = 0.0;
        
        // Visual effects
        private float _warningPulse = 0.5f;
        private bool _warningPulseDirection = true;
        private float _dangerFlash = 0.0f;
        private bool _dangerFlashDirection = true;
        private Color _currentWarningColor = Color.Orange;
        
        // Session times (reusing from sessionsGDI)
        private TimeSpan _nySessionStart = new TimeSpan(9, 30, 0);
        private TimeSpan _nySessionEnd = new TimeSpan(16, 0, 0);
        private TimeSpan _londonSessionStart = new TimeSpan(3, 0, 0);
        private TimeSpan _londonSessionEnd = new TimeSpan(11, 0, 0);
        private TimeSpan _tokyoSessionStart = new TimeSpan(19, 0, 0);
        private TimeSpan _tokyoSessionEnd = new TimeSpan(4, 0, 0);
        
        // Current active session
        private string _currentActiveSession = "None";
        
        // Custom fonts and brushes
        private Font _headerFont;
        private Font _sessionFont;
        private Font _warningFont;
        private Font _statsFont;
        
        #endregion
        
        #region Input Parameters
        
        [InputParameter("Daily Loss Limit ($)", 100)]
        public double DailyLossLimit { get; set; } = 500.0;
        
        [InputParameter("Warning Threshold ($)", 110)]
        public double WarningThreshold { get; set; } = 300.0;
        
        [InputParameter("Maximum Drawdown Limit ($)", 120)]
        public double MaxDrawdownLimit { get; set; } = 200.0;
        
        [InputParameter("Enable Audio Alerts", 200)]
        public bool EnableAudioAlerts { get; set; } = true;
        
        [InputParameter("Show Session Statistics", 210)]
        public bool ShowSessionStats { get; set; } = true;
        
        [InputParameter("Show Risk Panel", 220)]
        public bool ShowRiskPanel { get; set; } = true;
        
        [InputParameter("Show Performance Metrics", 230)]
        public bool ShowPerformanceMetrics { get; set; } = true;
        
        [InputParameter("Warning Color", 300)]
        public Color WarningColor { get; set; } = Color.Orange;
        
        [InputParameter("Danger Color", 310)]
        public Color DangerColor { get; set; } = Color.Red;
        
        [InputParameter("Safe Color", 320)]
        public Color SafeColor { get; set; } = Color.LimeGreen;
        
        [InputParameter("Panel Background", 330)]
        public Color PanelBackground { get; set; } = Color.FromArgb(180, 30, 30, 40);
        
        [InputParameter("Animation Speed", 400, 1, 10, 1, 0)]
        public int AnimationSpeed { get; set; } = 5;
        
        [InputParameter("Warning Sensitivity", 410, 1, 100, 1, 0)]
        public int WarningSensitivity { get; set; } = 50;
        
        [InputParameter("Reset Daily at Session Open", 500)]
        public bool ResetDailyAtSessionOpen { get; set; } = true;
        
        #endregion
        
        /// <summary>
        /// Constructor
        /// </summary>
        public SessionAngel() : base()
        {
            Name = "Session Angel";
            Description = "Trading Guardian & Risk Management with Session Statistics";
            
            // Add invisible line series for data storage
            AddLineSeries("PnL", Color.Transparent, 1, LineStyle.Solid);
            AddLineSeries("Drawdown", Color.Transparent, 1, LineStyle.Solid);
            AddLineSeries("Warning Level", Color.Transparent, 1, LineStyle.Solid);
            
            SeparateWindow = false;
        }
        
        /// <summary>
        /// Override ShortName property
        /// </summary>
        public override string ShortName => "SessionAngel";
        
        /// <summary>
        /// Initialize
        /// </summary>
        protected override void OnInit()
        {
            InitializeGraphicsResources();
            ResetSessionData();
            InitializeTradingTracking();
        }
        
        /// <summary>
        /// Initialize graphics resources
        /// </summary>
        private void InitializeGraphicsResources()
        {
            try
            {
                _headerFont?.Dispose();
                _sessionFont?.Dispose();
                _warningFont?.Dispose();
                _statsFont?.Dispose();
                
                _headerFont = new Font("Segoe UI", 18, FontStyle.Bold);
                _sessionFont = new Font("Segoe UI", 12, FontStyle.Regular);
                _warningFont = new Font("Segoe UI", 16, FontStyle.Bold);
                _statsFont = new Font("Consolas", 10, FontStyle.Regular);
            }
            catch
            {
                _headerFont = SystemFonts.DefaultFont;
                _sessionFont = SystemFonts.DefaultFont;
                _warningFont = SystemFonts.DefaultFont;
                _statsFont = SystemFonts.DefaultFont;
            }
        }
        
        /// <summary>
        /// Reset session data
        /// </summary>
        private void ResetSessionData()
        {
            _lastProcessedDate = DateTime.MinValue;
            _sessionStartTime = DateTime.Now;
            
            // Reset trading metrics
            _currentSessionPnL = 0.0;
            _maxDrawdown = 0.0;
            _maxProfit = 0.0;
            _mfe = 0.0;
            _mae = 0.0;
            
            // Reset warning states
            _warningActive = false;
            _dangerActive = false;
            _warningHistory.Clear();
            
            // Reset animation
            _animationFrame = 0;
            _warningPulse = 0.5f;
            _dangerFlash = 0.0f;
        }
        
        /// <summary>
        /// Initialize trading activity tracking
        /// </summary>
        private void InitializeTradingTracking()
        {
            // This would connect to account/position data
            // For now, we'll simulate with price movements
            _sessionStartBalance = 10000.0; // Default starting balance
        }
        
        /// <summary>
        /// Main update method
        /// </summary>
        protected override void OnUpdate(UpdateArgs args)
        {
            if (Count < 1) return;
            
            var currentTime = Time();
            var currentPrice = Close();
            
            // Reset if new trading day
            if (currentTime.Date != _lastProcessedDate.Date)
            {
                if (ResetDailyAtSessionOpen)
                {
                    _lastDayProfit = _currentSessionPnL;
                    ResetSessionData();
                }
                _lastProcessedDate = currentTime.Date;
            }
            
            // Update current active session
            UpdateActiveSession(currentTime);
            
            // Simulate trading activity (in real implementation, this would read actual account data)
            UpdateSimulatedTradingActivity(currentPrice);
            
            // Check risk thresholds
            CheckRiskThresholds();
            
            // Update statistics
            UpdateSessionStatistics();
            
            // Update animations
            UpdateAnimationEffects();
            
            // Set line series values
            SetLineSeries();
        }
        
        /// <summary>
        /// Update current active session
        /// </summary>
        private void UpdateActiveSession(DateTime currentTime)
        {
            var timeOfDay = currentTime.TimeOfDay;
            
            if (IsInSession(currentTime, _nySessionStart, _nySessionEnd))
                _currentActiveSession = "New York";
            else if (IsInSession(currentTime, _londonSessionStart, _londonSessionEnd))
                _currentActiveSession = "London";
            else if (IsInSession(currentTime, _tokyoSessionStart, _tokyoSessionEnd))
                _currentActiveSession = "Tokyo";
            else
                _currentActiveSession = "None";
        }
        
        /// <summary>
        /// Simulate trading activity (replace with real account data)
        /// </summary>
        private void UpdateSimulatedTradingActivity(double currentPrice)
        {
            // This is a placeholder - in real implementation, you would:
            // 1. Access actual account P&L
            // 2. Read position data
            // 3. Calculate real-time profit/loss
            
            // For demonstration, simulate P&L based on price volatility
            var priceChange = Count > 1 ? (currentPrice - Close(1)) : 0;
            var simulatedPnL = priceChange * 10; // Simulate position size effect
            
            _currentSessionPnL += simulatedPnL;
            
            // Update MFE/MAE
            if (_currentSessionPnL > _maxProfit)
            {
                _maxProfit = _currentSessionPnL;
                _mfe = _maxProfit;
            }
            
            if (_currentSessionPnL < _maxDrawdown)
            {
                _maxDrawdown = _currentSessionPnL;
                _mae = Math.Abs(_maxDrawdown);
            }
        }
        
        /// <summary>
        /// Check risk management thresholds
        /// </summary>
        private void CheckRiskThresholds()
        {
            var currentLoss = Math.Abs(Math.Min(_currentSessionPnL, 0));
            
            // Check danger threshold (daily loss limit)
            if (currentLoss >= DailyLossLimit)
            {
                if (!_dangerActive)
                {
                    _dangerActive = true;
                    TriggerDangerAlert();
                }
            }
            else
            {
                _dangerActive = false;
            }
            
            // Check warning threshold
            if (currentLoss >= WarningThreshold && !_dangerActive)
            {
                if (!_warningActive)
                {
                    _warningActive = true;
                    TriggerWarningAlert();
                }
            }
            else if (currentLoss < WarningThreshold * 0.8) // Hysteresis
            {
                _warningActive = false;
            }
            
            // Check drawdown limit
            if (Math.Abs(_maxDrawdown) >= MaxDrawdownLimit)
            {
                TriggerDrawdownAlert();
            }
        }
        
        /// <summary>
        /// Trigger warning alert
        /// </summary>
        private void TriggerWarningAlert()
        {
            _lastWarningTime = DateTime.Now;
            _warningHistory.Add(_lastWarningTime);
            
            if (EnableAudioAlerts)
            {
                // Audio alert would go here
                System.Console.Beep(800, 200);
            }
        }
        
        /// <summary>
        /// Trigger danger alert
        /// </summary>
        private void TriggerDangerAlert()
        {
            if (EnableAudioAlerts)
            {
                // More urgent audio alert
                for (int i = 0; i < 3; i++)
                {
                    System.Console.Beep(1000, 300);
                    System.Threading.Thread.Sleep(100);
                }
            }
        }
        
        /// <summary>
        /// Trigger drawdown alert
        /// </summary>
        private void TriggerDrawdownAlert()
        {
            if (EnableAudioAlerts)
            {
                System.Console.Beep(600, 500);
            }
        }
        
        /// <summary>
        /// Update session statistics
        /// </summary>
        private void UpdateSessionStatistics()
        {
            // Calculate win/loss ratios, profit factors, etc.
            // This would be expanded with real trading data
        }
        
        /// <summary>
        /// Update animation effects
        /// </summary>
        private void UpdateAnimationEffects()
        {
            var now = DateTime.Now;
            if ((now - _lastAnimationUpdate).TotalMilliseconds >= (50 / AnimationSpeed))
            {
                _animationFrame = (_animationFrame + 1) % 360;
                
                // Warning pulse effect
                if (_warningPulseDirection)
                {
                    _warningPulse += 0.03f * AnimationSpeed;
                    if (_warningPulse >= 1.0f)
                    {
                        _warningPulse = 1.0f;
                        _warningPulseDirection = false;
                    }
                }
                else
                {
                    _warningPulse -= 0.03f * AnimationSpeed;
                    if (_warningPulse <= 0.3f)
                    {
                        _warningPulse = 0.3f;
                        _warningPulseDirection = true;
                    }
                }
                
                // Danger flash effect
                if (_dangerActive)
                {
                    if (_dangerFlashDirection)
                    {
                        _dangerFlash += 0.1f * AnimationSpeed;
                        if (_dangerFlash >= 1.0f)
                        {
                            _dangerFlash = 1.0f;
                            _dangerFlashDirection = false;
                        }
                    }
                    else
                    {
                        _dangerFlash -= 0.1f * AnimationSpeed;
                        if (_dangerFlash <= 0.0f)
                        {
                            _dangerFlash = 0.0f;
                            _dangerFlashDirection = true;
                        }
                    }
                }
                
                _lastAnimationUpdate = now;
            }
        }
        
        /// <summary>
        /// Set line series values
        /// </summary>
        private void SetLineSeries()
        {
            SetValue(_currentSessionPnL, 0);
            SetValue(_maxDrawdown, 1);
            SetValue(_warningActive ? 1 : 0, 2);
        }
        
        /// <summary>
        /// Advanced painting with Session Angel graphics
        /// </summary>
        public override void OnPaintChart(PaintChartEventArgs args)
        {
            if (Count < 1) return;
            
            try
            {
                // Enable high-quality rendering
                args.Graphics.SmoothingMode = SmoothingMode.AntiAlias;
                args.Graphics.TextRenderingHint = TextRenderingHint.ClearTypeGridFit;
                args.Graphics.CompositingQuality = CompositingQuality.HighQuality;
                
                // Draw main angel panel
                DrawAngelPanel(args);
                
                // Draw risk status indicator
                DrawRiskStatusIndicator(args);
                
                // Draw session statistics
                if (ShowSessionStats)
                {
                    DrawSessionStatistics(args);
                }
                
                // Draw performance metrics
                if (ShowPerformanceMetrics)
                {
                    DrawPerformanceMetrics(args);
                }
                
                // Draw warning overlays
                DrawWarningOverlays(args);
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine($"SessionAngel drawing error: {ex.Message}");
            }
        }
        
        /// <summary>
        /// Draw main Session Angel panel
        /// </summary>
        private void DrawAngelPanel(PaintChartEventArgs args)
        {
            if (!ShowRiskPanel) return;
            
            var rect = args.Rectangle;
            var panelRect = new RectangleF(rect.X + 20, rect.Y + 20, 350, 200);
            
            // Draw panel background with gradient
            using (var panelBrush = new LinearGradientBrush(
                panelRect,
                PanelBackground,
                Color.FromArgb(120, PanelBackground),
                LinearGradientMode.Vertical))
            {
                args.Graphics.FillRoundRectangle(panelBrush, panelRect, 15);
            }
            
            // Draw panel border with angel glow
            var borderColor = _dangerActive ? DangerColor : 
                             _warningActive ? WarningColor : SafeColor;
            
            using (var borderPen = new Pen(Color.FromArgb((int)(200 * _warningPulse), borderColor), 2))
            {
                args.Graphics.DrawRoundRectangle(borderPen, panelRect, 15);
            }
            
            // Draw angel header
            var headerText = "👼 SESSION ANGEL";
            using (var headerBrush = new SolidBrush(Color.White))
            {
                args.Graphics.DrawString(headerText, _headerFont, headerBrush, 
                    panelRect.X + 15, panelRect.Y + 10);
            }
            
            // Draw status information
            var y = panelRect.Y + 45;
            var lineHeight = 18;
            
            DrawStatusLine(args, $"Active Session: {_currentActiveSession}", panelRect.X + 15, y, SafeColor);
            y += lineHeight;
            
            var pnlColor = _currentSessionPnL >= 0 ? SafeColor : DangerColor;
            DrawStatusLine(args, $"Session P&L: ${_currentSessionPnL:F2}", panelRect.X + 15, y, pnlColor);
            y += lineHeight;
            
            DrawStatusLine(args, $"Max Drawdown: ${_maxDrawdown:F2}", panelRect.X + 15, y, DangerColor);
            y += lineHeight;
            
            DrawStatusLine(args, $"Max Profit: ${_maxProfit:F2}", panelRect.X + 15, y, SafeColor);
            y += lineHeight;
            
            DrawStatusLine(args, $"Yesterday P&L: ${_lastDayProfit:F2}", panelRect.X + 15, y, 
                _lastDayProfit >= 0 ? SafeColor : DangerColor);
            y += lineHeight;
            
            // Draw risk levels
            y += 10;
            DrawStatusLine(args, $"Warning at: ${WarningThreshold:F0}", panelRect.X + 15, y, WarningColor);
            y += lineHeight;
            DrawStatusLine(args, $"Danger at: ${DailyLossLimit:F0}", panelRect.X + 15, y, DangerColor);
        }
        
        /// <summary>
        /// Draw status line with color
        /// </summary>
        private void DrawStatusLine(PaintChartEventArgs args, string text, float x, float y, Color color)
        {
            using (var brush = new SolidBrush(color))
            {
                args.Graphics.DrawString(text, _sessionFont, brush, x, y);
            }
        }
        
        /// <summary>
        /// Draw risk status indicator
        /// </summary>
        private void DrawRiskStatusIndicator(PaintChartEventArgs args)
        {
            var rect = args.Rectangle;
            var indicatorRect = new RectangleF(rect.Right - 100, rect.Y + 20, 70, 70);
            
            // Determine status color and text
            Color statusColor;
            string statusText;
            
            if (_dangerActive)
            {
                statusColor = Color.FromArgb((int)(255 * _dangerFlash), DangerColor);
                statusText = "DANGER";
            }
            else if (_warningActive)
            {
                statusColor = Color.FromArgb((int)(255 * _warningPulse), WarningColor);
                statusText = "WARNING";
            }
            else
            {
                statusColor = SafeColor;
                statusText = "SAFE";
            }
            
            // Draw status circle
            using (var statusBrush = new SolidBrush(Color.FromArgb(150, statusColor)))
            {
                args.Graphics.FillEllipse(statusBrush, indicatorRect);
            }
            
            using (var borderPen = new Pen(statusColor, 3))
            {
                args.Graphics.DrawEllipse(borderPen, indicatorRect);
            }
            
            // Draw status text
            var textRect = new RectangleF(indicatorRect.X, indicatorRect.Bottom + 5, indicatorRect.Width, 20);
            using (var textBrush = new SolidBrush(statusColor))
            {
                var format = new StringFormat { Alignment = StringAlignment.Center };
                args.Graphics.DrawString(statusText, _sessionFont, textBrush, textRect, format);
            }
        }
        
        /// <summary>
        /// Draw session statistics
        /// </summary>
        private void DrawSessionStatistics(PaintChartEventArgs args)
        {
            var rect = args.Rectangle;
            var statsRect = new RectangleF(rect.X + 20, rect.Y + 240, 300, 120);
            
            // Background
            using (var bgBrush = new SolidBrush(Color.FromArgb(160, 20, 20, 30)))
            {
                args.Graphics.FillRoundRectangle(bgBrush, statsRect, 10);
            }
            
            // Header
            using (var headerBrush = new SolidBrush(Color.LightBlue))
            {
                args.Graphics.DrawString("SESSION STATISTICS", _sessionFont, headerBrush,
                    statsRect.X + 10, statsRect.Y + 10);
            }
            
            var y = statsRect.Y + 35;
            var lineHeight = 16;
            
            using (var statsBrush = new SolidBrush(Color.LightGray))
            {
                args.Graphics.DrawString($"MFE (Max Favorable): ${_mfe:F2}", _statsFont, statsBrush, statsRect.X + 10, y);
                y += lineHeight;
                args.Graphics.DrawString($"MAE (Max Adverse): ${_mae:F2}", _statsFont, statsBrush, statsRect.X + 10, y);
                y += lineHeight;
                args.Graphics.DrawString($"Total Trades: {_totalTrades}", _statsFont, statsBrush, statsRect.X + 10, y);
                y += lineHeight;
                args.Graphics.DrawString($"Win Rate: {(_totalTrades > 0 ? (_winningTrades * 100.0 / _totalTrades):0):F1}%", 
                    _statsFont, statsBrush, statsRect.X + 10, y);
                y += lineHeight;
                args.Graphics.DrawString($"Largest Win: ${_largestWin:F2}", _statsFont, statsBrush, statsRect.X + 10, y);
                y += lineHeight;
                args.Graphics.DrawString($"Largest Loss: ${_largestLoss:F2}", _statsFont, statsBrush, statsRect.X + 10, y);
            }
        }
        
        /// <summary>
        /// Draw performance metrics
        /// </summary>
        private void DrawPerformanceMetrics(PaintChartEventArgs args)
        {
            var rect = args.Rectangle;
            var metricsRect = new RectangleF(rect.X + 340, rect.Y + 240, 250, 120);
            
            // Background
            using (var bgBrush = new SolidBrush(Color.FromArgb(160, 30, 20, 20)))
            {
                args.Graphics.FillRoundRectangle(bgBrush, metricsRect, 10);
            }
            
            // Header
            using (var headerBrush = new SolidBrush(Color.LightCoral))
            {
                args.Graphics.DrawString("PERFORMANCE", _sessionFont, headerBrush,
                    metricsRect.X + 10, metricsRect.Y + 10);
            }
            
            // Draw performance bars
            DrawPerformanceBar(args, "P&L", _currentSessionPnL, metricsRect.X + 10, metricsRect.Y + 40, 200, 15);
            DrawPerformanceBar(args, "Risk", -Math.Abs(_maxDrawdown), metricsRect.X + 10, metricsRect.Y + 65, 200, 15);
            DrawPerformanceBar(args, "Profit", _maxProfit, metricsRect.X + 10, metricsRect.Y + 90, 200, 15);
        }
        
        /// <summary>
        /// Draw performance bar
        /// </summary>
        private void DrawPerformanceBar(PaintChartEventArgs args, string label, double value, float x, float y, float width, float height)
        {
            // Background bar
            var barRect = new RectangleF(x + 50, y, width - 50, height);
            using (var bgBrush = new SolidBrush(Color.FromArgb(100, Color.Gray)))
            {
                args.Graphics.FillRectangle(bgBrush, barRect);
            }
            
            // Value bar
            var valuePercent = Math.Min(Math.Abs(value) / Math.Max(DailyLossLimit, 1), 1.0);
            var valueWidth = (float)(valuePercent * (width - 50));
            var valueRect = new RectangleF(x + 50, y, valueWidth, height);
            
            var barColor = value >= 0 ? SafeColor : (value <= -WarningThreshold ? DangerColor : WarningColor);
            using (var valueBrush = new SolidBrush(Color.FromArgb(180, barColor)))
            {
                args.Graphics.FillRectangle(valueBrush, valueRect);
            }
            
            // Label
            using (var labelBrush = new SolidBrush(Color.White))
            {
                args.Graphics.DrawString(label, _statsFont, labelBrush, x, y);
            }
        }
        
        /// <summary>
        /// Draw warning overlays
        /// </summary>
        private void DrawWarningOverlays(PaintChartEventArgs args)
        {
            if (!_warningActive && !_dangerActive) return;
            
            var rect = args.Rectangle;
            
            if (_dangerActive)
            {
                // Full screen danger overlay
                var overlayColor = Color.FromArgb((int)(50 * _dangerFlash), DangerColor);
                using (var overlayBrush = new SolidBrush(overlayColor))
                {
                    args.Graphics.FillRectangle(overlayBrush, rect);
                }
                
                // Danger message
                var dangerText = "⚠️ DAILY LOSS LIMIT REACHED! ⚠️";
                var textSize = args.Graphics.MeasureString(dangerText, _warningFont);
                var textX = rect.X + (rect.Width - textSize.Width) / 2;
                var textY = rect.Y + rect.Height / 2 - 50;
                
                using (var textBrush = new SolidBrush(Color.FromArgb((int)(255 * _dangerFlash), Color.White)))
                {
                    args.Graphics.DrawString(dangerText, _warningFont, textBrush, textX, textY);
                }
            }
            else if (_warningActive)
            {
                // Warning border
                var borderColor = Color.FromArgb((int)(100 * _warningPulse), WarningColor);
                using (var borderPen = new Pen(borderColor, 5))
                {
                    args.Graphics.DrawRectangle(borderPen, rect.X + 2, rect.Y + 2, rect.Width - 4, rect.Height - 4);
                }
            }
        }
        
        #region Helper Methods
        
        /// <summary>
        /// Check if current time is within a trading session
        /// </summary>
        public bool IsInSession(DateTime time, TimeSpan sessionStart, TimeSpan sessionEnd)
        {
            var timeOfDay = time.TimeOfDay;
            
            if (sessionEnd > sessionStart)
            {
                return timeOfDay >= sessionStart && timeOfDay <= sessionEnd;
            }
            else
            {
                return timeOfDay >= sessionStart || timeOfDay <= sessionEnd;
            }
        }
        
        /// <summary>
        /// Cleanup resources
        /// </summary>
        protected override void OnClear()
        {
            _headerFont?.Dispose();
            _sessionFont?.Dispose();
            _warningFont?.Dispose();
            _statsFont?.Dispose();
            base.OnClear();
        }
        
        #endregion
        
        #region Public Properties
        
        public double CurrentSessionPnL => _currentSessionPnL;
        public double MaxDrawdown => _maxDrawdown;
        public double MaxProfit => _maxProfit;
        public bool IsWarningActive => _warningActive;
        public bool IsDangerActive => _dangerActive;
        public string CurrentActiveSession => _currentActiveSession;
        public double MFE => _mfe;
        public double MAE => _mae;
        
        #endregion
    }
    
    #region Extension Methods (reusing from sessionsGDI)
    
    /// <summary>
    /// Extension methods for advanced graphics operations
    /// </summary>
    public static class SessionAngelGraphicsExtensions
    {
        /// <summary>
        /// Draw rounded rectangle
        /// </summary>
        public static void FillRoundRectangle(this Graphics graphics, Brush brush, RectangleF rect, float radius)
        {
            using (var path = GetRoundRectPath(rect, radius))
            {
                graphics.FillPath(brush, path);
            }
        }
        
        /// <summary>
        /// Draw rounded rectangle outline
        /// </summary>
        public static void DrawRoundRectangle(this Graphics graphics, Pen pen, RectangleF rect, float radius)
        {
            using (var path = GetRoundRectPath(rect, radius))
            {
                graphics.DrawPath(pen, path);
            }
        }
        
        /// <summary>
        /// Create rounded rectangle path
        /// </summary>
        private static GraphicsPath GetRoundRectPath(RectangleF rect, float radius)
        {
            var path = new GraphicsPath();
            radius = Math.Min(radius, Math.Min(rect.Width, rect.Height) / 2);
            
            path.AddArc(rect.X, rect.Y, radius * 2, radius * 2, 180, 90);
            path.AddArc(rect.Right - radius * 2, rect.Y, radius * 2, radius * 2, 270, 90);
            path.AddArc(rect.Right - radius * 2, rect.Bottom - radius * 2, radius * 2, radius * 2, 0, 90);
            path.AddArc(rect.X, rect.Bottom - radius * 2, radius * 2, radius * 2, 90, 90);
            path.CloseFigure();
            
            return path;
        }
    }
    
    #endregion
}
