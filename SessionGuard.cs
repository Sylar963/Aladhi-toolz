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
    /// Session Angel - Optimized Trading Guardian & Risk Management Indicator
    /// HFT-optimized version with reduced overhead and memory allocations
    /// </summary>
    public class OPAngelSession : Indicator
    {
        #region Private Fields - Optimized State Management

        // Session tracking - cached DateTime operations
        private DateTime _lastProcessedDate = DateTime.MinValue;
        private DateTime _sessionStartTime = DateTime.MinValue;
        private int _animationFrame = 0;
        private DateTime _cacheCurrentTime; // Cache current time per update cycle
        private long _lastAnimationTicks = 0; // Use ticks for faster comparison

        // Trading activity tracking - use doubles for better performance
        private double _sessionStartBalance = 0.0;
        private double _currentSessionPnL = 0.0;
        private double _maxDrawdown = 0.0;
        private double _maxProfit = 0.0;
        private double _lastDayProfit = 0.0;

        // Risk management - minimize bool operations
        private byte _riskState = 0; // 0=Safe, 1=Warning, 2=Danger (faster than multiple bools)
        private DateTime _lastWarningTime = DateTime.MinValue;
        private List<DateTime> _warningHistory = new List<DateTime>(16); // Pre-allocate capacity

        // Session statistics - pre-allocated
        private double _mfe = 0.0;
        private double _mae = 0.0;
        private int _totalTrades = 0;
        private int _winningTrades = 0;
        private int _losingTrades = 0;
        private double _largestWin = 0.0;
        private double _largestLoss = 0.0;

        // Visual effects - optimized animation state
        private float _warningPulse = 0.5f;
        private bool _warningPulseDirection = true;
        private float _dangerFlash = 0.0f;
        private bool _dangerFlashDirection = true;

        // Pre-calculated animation constants
        private readonly float _pulseIncrement = 0.03f;
        private readonly float _flashIncrement = 0.1f;
        private readonly float _pulseMin = 0.3f;
        private readonly float _pulseMax = 1.0f;
        private readonly float _warningHysteresis = 0.8f;

        // Session times - readonly for performance
        private readonly TimeSpan _nySessionStart = new TimeSpan(9, 30, 0);
        private readonly TimeSpan _nySessionEnd = new TimeSpan(16, 0, 0);
        private readonly TimeSpan _londonSessionStart = new TimeSpan(3, 0, 0);
        private readonly TimeSpan _londonSessionEnd = new TimeSpan(11, 0, 0);
        private readonly TimeSpan _tokyoSessionStart = new TimeSpan(19, 0, 0);
        private readonly TimeSpan _tokyoSessionEnd = new TimeSpan(4, 0, 0);

        // Current active session - use byte for memory efficiency
        private byte _currentActiveSessionId = 0; // 0=None, 1=NY, 2=London, 3=Tokyo

        // Cached session names - avoid string allocations
        private static readonly string[] _sessionNames = { "None", "New York", "London", "Tokyo" };

        // Graphics objects cache - reuse instead of creating
        private Font _headerFont;
        private Font _sessionFont;
        private Font _warningFont;
        private Font _statsFont;

        // Cached brushes and pens - avoid repeated allocations
        private SolidBrush _cachedSafeBrush;
        private SolidBrush _cachedWarningBrush;
        private SolidBrush _cachedDangerBrush;
        private SolidBrush _cachedWhiteBrush;
        private SolidBrush _cachedGrayBrush;

        // Pre-calculated values to avoid repeated calculations
        private double _cachedCurrentLoss = 0.0;
        private bool _needsRiskRecalc = true;

        // Animation timing optimization
        private readonly long _animationIntervalTicks;

        // PERFORMANCE: Rate limit account data access
        private int _updateCounter = 0;
        private const int ACCOUNT_UPDATE_INTERVAL = 50; // Only check account every 50 updates

        // Volume bar specific tracking
        private double _currentVolumeBarVolume = 0.0;
        private double _averageVolumePerBar = 0.0;
        private double _totalVolumeProcessed = 0.0;
        private int _volumeBarCount = 0;

        // Position transition tracking for trade stats and risk reset
        private bool _wasInPosition = false;
        private double _lastFlatBalance = 0.0;

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
        public bool ShowRiskPanel { get; set; } = true; // Show panel by default

        [InputParameter("Show Performance Metrics", 230)]
        public bool ShowPerformanceMetrics { get; set; } = false; // PERFORMANCE: Default OFF

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

        [InputParameter("Volume Bar Size", 510, 100, 10000, 100, 0)]
        public int VolumeBarSize { get; set; } = 1000;

        [InputParameter("Show Volume Statistics", 520)]
        public bool ShowVolumeStats { get; set; } = true;

        [InputParameter("Account Name (leave empty for auto)", 530)]
        public string AccountName { get; set; } = ""; // Empty = auto select first

        // Simple account tracking
        private string _currentAccountName = "";
        private string _availableAccountsList = "";

        #endregion

        /// <summary>
        /// Constructor - Volume-based Guardian Angel
        /// </summary>
        public OPAngelSession() : base()
        {
            Name = "Volume Guard Angel";
            Description = "Volume-based Trading Guardian & Risk Management";

            // Pre-calculate animation interval in ticks for faster comparison
            _animationIntervalTicks = TimeSpan.FromMilliseconds(50).Ticks;

            // Add invisible line series for data storage
            AddLineSeries("PnL", Color.Transparent, 1, LineStyle.Solid);
            AddLineSeries("Drawdown", Color.Transparent, 1, LineStyle.Solid);
            AddLineSeries("Warning Level", Color.Transparent, 1, LineStyle.Solid);

            SeparateWindow = true; // PERFORMANCE: Move to separate window

            // Note: Volume aggregation will be handled through chart settings, not in indicator code
        }

        /// <summary>
        /// Override ShortName property
        /// </summary>
        public override string ShortName => "VolumeGuard";

        /// <summary>
        /// Initialize - Create cached objects and detect accounts
        /// </summary>
        protected override void OnInit()
        {
            InitializeOptimizedGraphicsResources();
            DetectAvailableAccounts();
            ResetSessionData();
            InitializeTradingTracking();
        }

        /// <summary>
        /// Initialize graphics resources - cache frequently used objects
        /// </summary>
        private void InitializeOptimizedGraphicsResources()
        {
            try
            {
                // Dispose existing resources
                DisposeGraphicsResources();

                // Create fonts
                _headerFont = new Font("Segoe UI", 18, FontStyle.Bold);
                _sessionFont = new Font("Segoe UI", 12, FontStyle.Regular);
                _warningFont = new Font("Segoe UI", 16, FontStyle.Bold);
                _statsFont = new Font("Consolas", 10, FontStyle.Regular);

                // Create cached brushes to avoid repeated allocations
                _cachedSafeBrush = new SolidBrush(SafeColor);
                _cachedWarningBrush = new SolidBrush(WarningColor);
                _cachedDangerBrush = new SolidBrush(DangerColor);
                _cachedWhiteBrush = new SolidBrush(Color.White);
                _cachedGrayBrush = new SolidBrush(Color.LightGray);
            }
            catch
            {
                // Fallback to system fonts if creation fails
                _headerFont = SystemFonts.DefaultFont;
                _sessionFont = SystemFonts.DefaultFont;
                _warningFont = SystemFonts.DefaultFont;
                _statsFont = SystemFonts.DefaultFont;

                // Create basic brushes
                _cachedSafeBrush = new SolidBrush(Color.Green);
                _cachedWarningBrush = new SolidBrush(Color.Orange);
                _cachedDangerBrush = new SolidBrush(Color.Red);
                _cachedWhiteBrush = new SolidBrush(Color.White);
                _cachedGrayBrush = new SolidBrush(Color.Gray);
            }
        }

        /// <summary>
        /// Reset session data - FIXED: Reset session baseline properly
        /// </summary>
        private void ResetSessionData()
        {
            _lastProcessedDate = DateTime.MinValue;
            _sessionStartTime = DateTime.Now;

            // Reset trading metrics - FIXED: Include session start balance reset
            _sessionStartBalance = 0.0; // CRITICAL FIX: Reset baseline to trigger new session
            _currentSessionPnL = 0.0;
            _maxDrawdown = 0.0;
            _maxProfit = 0.0;
            _mfe = 0.0;
            _mae = 0.0;

            // Reset risk state
            _riskState = 0; // Safe
            _warningHistory.Clear();
            _needsRiskRecalc = true;

            // Reset animation
            _animationFrame = 0;
            _warningPulse = 0.5f;
            _dangerFlash = 0.0f;
            _lastAnimationTicks = 0;

            // Reset volume tracking
            _currentVolumeBarVolume = 0.0;
            _averageVolumePerBar = 0.0;
            _totalVolumeProcessed = 0.0;
            _volumeBarCount = 0;

            // Reset position-tracking baselines
            _wasInPosition = false;
            _lastFlatBalance = 0.0;
        }

        /// <summary>
        /// CLEAN LOGIC: Detect all available accounts
        /// </summary>
        private void DetectAvailableAccounts()
        {
            try
            {
                var accounts = Core.Instance?.Accounts;
                if (accounts != null && accounts.Any())
                {
                    var accountNames = new List<string>();

                    foreach (var account in accounts)
                    {
                        var name = account.Name ?? "Unknown";
                        accountNames.Add(name);
                    }

                    _availableAccountsList = string.Join(", ", accountNames);
                    System.Diagnostics.Debug.WriteLine($"Volume Guard Angel: Available accounts: {_availableAccountsList}");
                }
                else
                {
                    _availableAccountsList = "No accounts found";
                    System.Diagnostics.Debug.WriteLine("Volume Guard Angel: No accounts detected");
                }
            }
            catch (Exception ex)
            {
                _availableAccountsList = "Error detecting accounts";
                System.Diagnostics.Debug.WriteLine($"Volume Guard Angel: Error detecting accounts: {ex.Message}");
            }
        }

        /// <summary>
        /// CLEAN LOGIC: Get selected account by name or auto-select first
        /// </summary>
        private Account GetSelectedAccount()
        {
            try
            {
                var accounts = Core.Instance?.Accounts;
                if (accounts == null || !accounts.Any())
                    return null;

                // If user specified account name, find it
                if (!string.IsNullOrEmpty(AccountName))
                {
                    var selectedAccount = accounts.FirstOrDefault(a =>
                        a.Name?.Equals(AccountName, StringComparison.OrdinalIgnoreCase) == true ||
                        a.Name?.Contains(AccountName, StringComparison.OrdinalIgnoreCase) == true);

                    if (selectedAccount != null)
                    {
                        _currentAccountName = selectedAccount.Name ?? "Unknown";
                        return selectedAccount;
                    }
                }

                // Auto-select first account
                var firstAccount = accounts.FirstOrDefault();
                _currentAccountName = firstAccount?.Name ?? "Unknown";
                return firstAccount;
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine($"Volume Guard Angel: Error getting account: {ex.Message}");
                return null;
            }
        }

        /// <summary>
        /// Initialize trading tracking - FIXED: No fake balance
        /// </summary>
        private void InitializeTradingTracking()
        {
            // FIXED: No hard-coded fake balance - let it initialize from real account data
            _sessionStartBalance = 0.0; // Will be set from real account equity on first update
        }

        /// <summary>
        /// Main update method - PERFORMANCE CRITICAL: Rate limited
        /// </summary>
        protected override void OnUpdate(UpdateArgs args)
        {
            if (Count < 1) return;

            _updateCounter++;

            // PERFORMANCE: Only do expensive operations every Nth update
            if (_updateCounter % ACCOUNT_UPDATE_INTERVAL == 0)
            {
                // Cache current time once per update cycle
                _cacheCurrentTime = Time();

                // Fast date comparison - check if new trading day
                if (_cacheCurrentTime.Date != _lastProcessedDate.Date)
                {
                    if (ResetDailyAtSessionOpen)
                    {
                        // FIXED: Save yesterday's P&L BEFORE reset
                        _lastDayProfit = _currentSessionPnL;
                        ResetSessionData();
                    }
                    _lastProcessedDate = _cacheCurrentTime.Date;
                }

                // Update current active session - optimized
                UpdateActiveSessionOptimized();

                // Update trading activity - EXPENSIVE! Rate limited
                UpdateTradingActivityOptimized();

                // Check risk thresholds - only when needed
                if (_needsRiskRecalc)
                {
                    CheckRiskThresholdsOptimized();
                    _needsRiskRecalc = false;
                }
            }

            // FAST PATH: Only do lightweight operations every update
            SetLineSeriesOptimized();
        }

        /// <summary>
        /// Update active session - optimized with byte comparison
        /// </summary>
        private void UpdateActiveSessionOptimized()
        {
            var timeOfDay = _cacheCurrentTime.TimeOfDay;

            byte newSessionId = 0; // None

            if (IsInSessionFast(timeOfDay, _nySessionStart, _nySessionEnd))
                newSessionId = 1; // NY
            else if (IsInSessionFast(timeOfDay, _londonSessionStart, _londonSessionEnd))
                newSessionId = 2; // London
            else if (IsInSessionFast(timeOfDay, _tokyoSessionStart, _tokyoSessionEnd))
                newSessionId = 3; // Tokyo

            _currentActiveSessionId = newSessionId;
        }

        /// <summary>
        /// Fast session check - optimized comparison
        /// </summary>
        private bool IsInSessionFast(TimeSpan timeOfDay, TimeSpan sessionStart, TimeSpan sessionEnd)
        {
            return sessionEnd > sessionStart ?
                (timeOfDay >= sessionStart && timeOfDay <= sessionEnd) :
                (timeOfDay >= sessionStart || timeOfDay <= sessionEnd);
        }

        /// <summary>
        /// Update trading activity - REAL ACCOUNT DATA with dropdown selection!
        /// </summary>
        private void UpdateTradingActivityOptimized()
        {
            // GET REAL ACCOUNT DATA - User selected account from dropdown!
            try
            {
                var account = GetSelectedAccount();

                if (account != null)
                {
                    // REAL P&L from YOUR SELECTED account
                    var currentBalance = account.Balance;

                    // Calculate total unrealized P&L from positions for THIS account only
                    var totalUnrealizedPnL = 0.0;
                    var inPosition = false;
                    foreach (Position pos in Core.Positions)
                    {
                        if (pos.ConnectionId == account.ConnectionId)
                        {
                            totalUnrealizedPnL += pos.NetPnL.Value;
                            try
                            {
                                if (pos.Quantity != 0)
                                    inPosition = true;
                            }
                            catch { /* Quantity property may not be available in some connections */ }
                        }
                    }

                    var currentEquity = currentBalance + totalUnrealizedPnL;

                    // FIXED: Track session P&L based on REAL account data
                    if (_sessionStartBalance == 0.0)
                    {
                        _sessionStartBalance = currentEquity; // Set fresh baseline for new session
                        _currentSessionPnL = 0.0; // Start at zero for new session
                    }
                    else
                    {
                        _currentSessionPnL = currentEquity - _sessionStartBalance; // REAL session P&L!
                    }
                    // Risk only active while holding a position; turn off when flat
                    if (!inPosition)
                    {
                        if (_wasInPosition)
                        {
                            // Position just closed: compute realized PnL using balance delta
                            var realizedTradePnL = account.Balance - _lastFlatBalance;
                            _totalTrades++;
                            if (realizedTradePnL > 0) _winningTrades++;
                            else if (realizedTradePnL < 0) _losingTrades++;
                            if (realizedTradePnL > _largestWin) _largestWin = realizedTradePnL;
                            if (realizedTradePnL < _largestLoss) _largestLoss = realizedTradePnL;
                        }

                        _wasInPosition = false;
                        _riskState = 0;            // Shut down warning/danger when flat
                        _needsRiskRecalc = false;  // Skip risk calc while flat
                    }
                    else
                    {
                        if (!_wasInPosition)
                        {
                            // Position just opened: capture balance baseline for realized PnL
                            _lastFlatBalance = account.Balance;
                        }
                        _wasInPosition = true;
                        _needsRiskRecalc = true;
                    }
                }
                else
                {
                    System.Diagnostics.Debug.WriteLine("Volume Guard Angel: No account selected or found");
                }
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine($"Volume Guard Angel: Account access error: {ex.Message}");
            }

            // FIXED: Update MFE/MAE based on REAL P&L data
            if (_currentSessionPnL > _maxProfit)
            {
                _maxProfit = _currentSessionPnL;
                _mfe = _maxProfit;
            }

            if (_currentSessionPnL < _maxDrawdown)
            {
                _maxDrawdown = _currentSessionPnL;
                _mae = Math.Abs(_maxDrawdown); // FIXED: Use proper absolute value for MAE
            }

            // Track volume-based statistics
            if (Count > 0)
            {
                _volumeBarCount = Count;
                _currentVolumeBarVolume = Volume();
                _totalVolumeProcessed += _currentVolumeBarVolume;
                _averageVolumePerBar = _totalVolumeProcessed / _volumeBarCount;

                // Trade statistics will be updated based on real account data
                // No simulation-based statistics
            }
        }

        /// <summary>
        /// Check risk thresholds - optimized logic
        /// </summary>
        private void CheckRiskThresholdsOptimized()
        {
            // Pre-calculate current loss once
            _cachedCurrentLoss = _currentSessionPnL < 0.0 ? -_currentSessionPnL : 0.0;

            var previousRiskState = _riskState;

            // Optimized state machine
            if (_cachedCurrentLoss >= DailyLossLimit)
            {
                _riskState = 2; // Danger
            }
            else if (_cachedCurrentLoss >= WarningThreshold)
            {
                _riskState = 1; // Warning
            }
            else if (_cachedCurrentLoss < WarningThreshold * _warningHysteresis)
            {
                _riskState = 0; // Safe
            }

            // Trigger alerts only on state change
            if (_riskState != previousRiskState)
            {
                switch (_riskState)
                {
                    case 1: TriggerWarningAlertOptimized(); break;
                    case 2: TriggerDangerAlertOptimized(); break;
                }
            }

            // Check drawdown limit
            if (-_maxDrawdown >= MaxDrawdownLimit)
            {
                TriggerDrawdownAlertOptimized();
            }
        }

        /// <summary>
        /// Optimized warning alert
        /// </summary>
        private void TriggerWarningAlertOptimized()
        {
            _lastWarningTime = _cacheCurrentTime;
            _warningHistory.Add(_lastWarningTime);

            if (EnableAudioAlerts)
            {
                System.Console.Beep(800, 200);
            }
        }

        /// <summary>
        /// Optimized danger alert
        /// </summary>
        private void TriggerDangerAlertOptimized()
        {
            if (EnableAudioAlerts)
            {
                // Reduced audio alert for performance
                System.Console.Beep(1000, 300);
            }
        }

        /// <summary>
        /// Optimized drawdown alert
        /// </summary>
        private void TriggerDrawdownAlertOptimized()
        {
            if (EnableAudioAlerts)
            {
                System.Console.Beep(600, 500);
            }
        }

        /// <summary>
        /// Update session statistics - placeholder
        /// </summary>
        private void UpdateSessionStatistics()
        {
            // Placeholder for future implementation
        }

        /// <summary>
        /// Update animation effects - rate limited and optimized
        /// </summary>
        private void UpdateAnimationEffectsOptimized()
        {
            var currentTicks = _cacheCurrentTime.Ticks;

            // Rate-limited animation updates using ticks for faster comparison
            if ((currentTicks - _lastAnimationTicks) >= (_animationIntervalTicks / AnimationSpeed))
            {
                _animationFrame = (_animationFrame + 1) % 360;

                // Optimized pulse calculation
                var pulseSpeed = _pulseIncrement * AnimationSpeed;

                if (_warningPulseDirection)
                {
                    _warningPulse += pulseSpeed;
                    if (_warningPulse >= _pulseMax)
                    {
                        _warningPulse = _pulseMax;
                        _warningPulseDirection = false;
                    }
                }
                else
                {
                    _warningPulse -= pulseSpeed;
                    if (_warningPulse <= _pulseMin)
                    {
                        _warningPulse = _pulseMin;
                        _warningPulseDirection = true;
                    }
                }

                // Optimized danger flash (only when in danger state)
                if (_riskState == 2) // Danger
                {
                    var flashSpeed = _flashIncrement * AnimationSpeed;

                    if (_dangerFlashDirection)
                    {
                        _dangerFlash += flashSpeed;
                        if (_dangerFlash >= 1.0f)
                        {
                            _dangerFlash = 1.0f;
                            _dangerFlashDirection = false;
                        }
                    }
                    else
                    {
                        _dangerFlash -= flashSpeed;
                        if (_dangerFlash <= 0.0f)
                        {
                            _dangerFlash = 0.0f;
                            _dangerFlashDirection = true;
                        }
                    }
                }

                _lastAnimationTicks = currentTicks;
            }
        }

        /// <summary>
        /// Set line series values - optimized
        /// </summary>
        private void SetLineSeriesOptimized()
        {
            SetValue(_currentSessionPnL, 0);
            SetValue(_maxDrawdown, 1);
            SetValue(_riskState > 0 ? 1.0 : 0.0, 2); // Convert risk state to double
        }

        // Volume bar tracking
        private int _lastVolumeBarCount = 0;
        private bool _newVolumeBarDetected = false;
        private byte _lastPaintedRiskState = 255; // Force initial paint
        private bool _forceRepaint = false;

        /// <summary>
        /// Helper method to check if enough time has passed for updates
        /// </summary>
        private bool ShouldUpdateComponent(long intervalMultiplier = 1)
        {
            return (DateTime.Now.Ticks - _lastAnimationTicks) >= (_animationIntervalTicks * intervalMultiplier);
        }

        /// <summary>
        /// Helper method to create alpha-blended colors
        /// </summary>
        private Color CreateAlphaColor(int alpha, Color baseColor)
        {
            return Color.FromArgb(alpha, baseColor);
        }

        /// <summary>
        /// Helper method to create animated alpha colors
        /// </summary>
        private Color CreateAnimatedColor(float intensity, Color baseColor)
        {
            return Color.FromArgb((int)(255 * intensity), baseColor);
        }

        /// <summary>
        /// Helper method to draw background panels with consistent styling
        /// </summary>
        private void DrawBackgroundPanel(Graphics graphics, RectangleF rect, Color baseColor, int alpha = 160)
        {
            using (var bgBrush = new SolidBrush(CreateAlphaColor(alpha, baseColor)))
            {
                DrawRoundedRectangle(graphics, bgBrush, rect, 10);
            }
        }

        /// <summary>
        /// Volume-based painting - Updates on new volume bars OR risk state changes!
        /// </summary>
        public override void OnPaintChart(PaintChartEventArgs args)
        {
            if (Count < 1) return;

            // Check for new volume bar
            _newVolumeBarDetected = (Count != _lastVolumeBarCount);

            // Check for risk state change (CRITICAL: Always show risk warnings!)
            var riskStateChanged = (_riskState != _lastPaintedRiskState);

            // Force repaint for animations when in warning/danger states
            var needsAnimationUpdate = (_riskState > 0) && ShouldUpdateComponent(1 / AnimationSpeed);

            // Force repaint for UI components when enabled (slower updates for performance)
            var needsPerformanceUpdate = ShowPerformanceMetrics && ShouldUpdateComponent(2);
            var needsStatsUpdate = ShowSessionStats && ShouldUpdateComponent(2);
            var needsRiskPanelUpdate = ShowRiskPanel && ShouldUpdateComponent(2);

            // Force repaint if any display component needs updating
            _forceRepaint = _newVolumeBarDetected || riskStateChanged || needsAnimationUpdate ||
                           needsPerformanceUpdate || needsStatsUpdate || needsRiskPanelUpdate;

            if (!_forceRepaint)
                return; // No changes, skip graphics

            _lastVolumeBarCount = Count;
            _lastPaintedRiskState = _riskState;

            try
            {
                // Since volume bars are infrequent, we can afford HIGH QUALITY graphics!
                args.Graphics.SmoothingMode = SmoothingMode.AntiAlias;
                args.Graphics.TextRenderingHint = TextRenderingHint.ClearTypeGridFit;
                args.Graphics.CompositingQuality = CompositingQuality.HighQuality;

                // Draw components using cached objects
                DrawAngelPanelOptimized(args);
                DrawRiskStatusIndicatorOptimized(args);

                if (ShowSessionStats)
                {
                    DrawSessionStatisticsOptimized(args);
                }

                // Volume stats removed - using real account data only

                if (ShowPerformanceMetrics)
                {
                    DrawPerformanceMetricsOptimized(args);
                }

                DrawWarningOverlaysOptimized(args);
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine($"OPAngelSession drawing error: {ex.Message}");
            }
        }

        /// <summary>
        /// Draw main angel panel - optimized with cached objects
        /// </summary>
        private void DrawAngelPanelOptimized(PaintChartEventArgs args)
        {
            if (!ShowRiskPanel) return;

            var rect = args.Rectangle;
            var panelRect = new RectangleF(rect.X + 20, rect.Y + 20, 350, 200);

            // Use cached brushes and optimized gradient
            using (var panelBrush = new LinearGradientBrush(
                panelRect,
                PanelBackground,
                Color.FromArgb(120, PanelBackground),
                LinearGradientMode.Vertical))
            {
                DrawRoundedRectangle(args.Graphics, panelBrush, panelRect, 15);
            }

            // Optimized border color selection
            Color borderColor;
            switch (_riskState)
            {
                case 2: borderColor = DangerColor; break;
                case 1: borderColor = WarningColor; break;
                default: borderColor = SafeColor; break;
            }

            using (var borderPen = new Pen(CreateAnimatedColor(_warningPulse * 200 / 255f, borderColor), 2))
            {
                DrawRoundedRectangleBorder(args.Graphics, borderPen, panelRect, 15);
            }

            // Draw header using cached brush
            args.Graphics.DrawString("👼 VOLUME GUARD ANGEL", _headerFont, _cachedWhiteBrush,
                panelRect.X + 15, panelRect.Y + 10);

            // Draw status lines - optimized layout
            var y = panelRect.Y + 45;
            const float lineHeight = 18;

            // Show which account is being monitored - CLEAN DISPLAY
            var accountDisplay = string.IsNullOrEmpty(_currentAccountName) ? "Auto-selecting..." : _currentAccountName;
            DrawStatusLineOptimized(args, $"Account: {accountDisplay}",
                panelRect.X + 15, y, _cachedSafeBrush);
            y += lineHeight;

            // Show available accounts for reference
            if (!string.IsNullOrEmpty(_availableAccountsList))
            {
                DrawStatusLineOptimized(args, $"Available: {_availableAccountsList}",
                    panelRect.X + 15, y, _cachedGrayBrush);
                y += lineHeight;
            }

            // Use cached session name and optimized drawing
            DrawStatusLineOptimized(args, $"Active Session: {_sessionNames[_currentActiveSessionId]}",
                panelRect.X + 15, y, _cachedSafeBrush);
            y += lineHeight;

            var pnlBrush = _currentSessionPnL >= 0 ? _cachedSafeBrush : _cachedDangerBrush;
            DrawStatusLineOptimized(args, $"Session P&L: ${_currentSessionPnL:F2}",
                panelRect.X + 15, y, pnlBrush);
            y += lineHeight;

            DrawStatusLineOptimized(args, $"Max Drawdown: ${_maxDrawdown:F2}",
                panelRect.X + 15, y, _cachedDangerBrush);
            y += lineHeight;

            DrawStatusLineOptimized(args, $"Max Profit: ${_maxProfit:F2}",
                panelRect.X + 15, y, _cachedSafeBrush);
            y += lineHeight;

            var lastDayBrush = _lastDayProfit >= 0 ? _cachedSafeBrush : _cachedDangerBrush;
            DrawStatusLineOptimized(args, $"Yesterday P&L: ${_lastDayProfit:F2}",
                panelRect.X + 15, y, lastDayBrush);
            y += lineHeight;

            // Draw volume information
            y += 10;
            DrawStatusLineOptimized(args, $"Volume Bars: {_volumeBarCount} | Size: {VolumeBarSize}",
                panelRect.X + 15, y, _cachedGrayBrush);
            y += lineHeight;
            DrawStatusLineOptimized(args, $"Current Volume: {_currentVolumeBarVolume:F0}",
                panelRect.X + 15, y, _cachedGrayBrush);
            y += lineHeight;

            // Draw risk levels
            DrawStatusLineOptimized(args, $"Warning at: ${WarningThreshold:F0}",
                panelRect.X + 15, y, _cachedWarningBrush);
            y += lineHeight;
            DrawStatusLineOptimized(args, $"Danger at: ${DailyLossLimit:F0}",
                panelRect.X + 15, y, _cachedDangerBrush);
        }

        /// <summary>
        /// Draw status line - optimized with cached brush
        /// </summary>
        private void DrawStatusLineOptimized(PaintChartEventArgs args, string text, float x, float y, SolidBrush brush)
        {
            args.Graphics.DrawString(text, _sessionFont, brush, x, y);
        }

        /// <summary>
        /// Draw risk status indicator - optimized
        /// </summary>
        private void DrawRiskStatusIndicatorOptimized(PaintChartEventArgs args)
        {
            var rect = args.Rectangle;
            var indicatorRect = new RectangleF(rect.Right - 100, rect.Y + 20, 70, 70);

            // Optimized status determination
            Color statusColor;
            string statusText;
            float intensity;

            switch (_riskState)
            {
                case 2: // Danger
                    statusColor = CreateAnimatedColor(_dangerFlash, DangerColor);
                    statusText = "DANGER";
                    intensity = _dangerFlash;
                    break;
                case 1: // Warning
                    statusColor = CreateAnimatedColor(_warningPulse, WarningColor);
                    statusText = "WARNING";
                    intensity = _warningPulse;
                    break;
                default: // Safe
                    statusColor = SafeColor;
                    statusText = "SAFE";
                    intensity = 1.0f;
                    break;
            }

            // Draw status circle
            using (var statusBrush = new SolidBrush(CreateAlphaColor(150, statusColor)))
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
        /// Draw session statistics - MOVED 40% more contrary to Volume Guard Angel
        /// </summary>
        private void DrawSessionStatisticsOptimized(PaintChartEventArgs args)
        {
            var rect = args.Rectangle;
            // MOVED: Volume Guard Angel is at top-left (X+20, Y+20), so move stats 40% more towards bottom-right
            var statsRect = new RectangleF(rect.Right - 820, rect.Bottom - 200, 300, 120);

            // Background
            DrawBackgroundPanel(args.Graphics, statsRect, Color.FromArgb(20, 20, 30));

            // Header
            using (var headerBrush = new SolidBrush(Color.LightBlue))
            {
                args.Graphics.DrawString("SESSION STATISTICS", _sessionFont, headerBrush,
                    statsRect.X + 10, statsRect.Y + 10);
            }

            // Statistics content - optimized layout
            var y = statsRect.Y + 35;
            const float lineHeight = 16;

            args.Graphics.DrawString($"MFE (Max Favorable): ${_mfe:F2}", _statsFont, _cachedGrayBrush, statsRect.X + 10, y);
            y += lineHeight;
            args.Graphics.DrawString($"MAE (Max Adverse): ${_mae:F2}", _statsFont, _cachedGrayBrush, statsRect.X + 10, y);
            y += lineHeight;
            args.Graphics.DrawString($"Total Trades: {_totalTrades}", _statsFont, _cachedGrayBrush, statsRect.X + 10, y);
            y += lineHeight;

            var winRate = _totalTrades > 0 ? (_winningTrades * 100.0 / _totalTrades) : 0.0;
            args.Graphics.DrawString($"Win Rate: {winRate:F1}%", _statsFont, _cachedGrayBrush, statsRect.X + 10, y);
            y += lineHeight;
            args.Graphics.DrawString($"Largest Win: ${_largestWin:F2}", _statsFont, _cachedGrayBrush, statsRect.X + 10, y);
            y += lineHeight;
            args.Graphics.DrawString($"Largest Loss: ${_largestLoss:F2}", _statsFont, _cachedGrayBrush, statsRect.X + 10, y);
        }

        /// <summary>
        /// Draw performance metrics - optimized
        /// </summary>
        private void DrawPerformanceMetricsOptimized(PaintChartEventArgs args)
        {
            var rect = args.Rectangle;
            var metricsRect = new RectangleF(rect.X + 340, rect.Y + 240, 250, 120);

            // Background
            DrawBackgroundPanel(args.Graphics, metricsRect, Color.FromArgb(30, 20, 20));

            // Header
            using (var headerBrush = new SolidBrush(Color.LightCoral))
            {
                args.Graphics.DrawString("PERFORMANCE", _sessionFont, headerBrush,
                    metricsRect.X + 10, metricsRect.Y + 10);
            }

            // Performance bars - optimized drawing
            DrawPerformanceBarOptimized(args, "P&L", _currentSessionPnL, metricsRect.X + 10, metricsRect.Y + 40, 200, 15);
            DrawPerformanceBarOptimized(args, "Risk", -Math.Abs(_maxDrawdown), metricsRect.X + 10, metricsRect.Y + 65, 200, 15);
            DrawPerformanceBarOptimized(args, "Profit", _maxProfit, metricsRect.X + 10, metricsRect.Y + 90, 200, 15);
        }

        /// <summary>
        /// Draw performance bar - optimized calculations
        /// </summary>
        private void DrawPerformanceBarOptimized(PaintChartEventArgs args, string label, double value, float x, float y, float width, float height)
        {
            // Background bar
            var barRect = new RectangleF(x + 50, y, width - 50, height);
            using (var bgBrush = new SolidBrush(CreateAlphaColor(100, Color.Gray)))
            {
                args.Graphics.FillRectangle(bgBrush, barRect);
            }

            // Value bar - optimized calculations
            var denominator = Math.Max(DailyLossLimit, 1.0);
            var valuePercent = Math.Min(Math.Abs(value) / denominator, 1.0);
            var valueWidth = (float)(valuePercent * (width - 50));
            var valueRect = new RectangleF(x + 50, y, valueWidth, height);

            // Optimized color selection
            Color barColor = value >= 0 ? SafeColor :
                           (value <= -WarningThreshold ? DangerColor : WarningColor);

            using (var valueBrush = new SolidBrush(CreateAlphaColor(180, barColor)))
            {
                args.Graphics.FillRectangle(valueBrush, valueRect);
            }

            // Label
            args.Graphics.DrawString(label, _statsFont, _cachedWhiteBrush, x, y);
        }

        /// <summary>
        /// Draw warning overlays - optimized rendering
        /// </summary>
        private void DrawWarningOverlaysOptimized(PaintChartEventArgs args)
        {
            if (_riskState == 0) return; // Safe - no overlays needed

            var rect = args.Rectangle;

            if (_riskState == 2) // Danger
            {
                // Full screen danger overlay
                var overlayColor = CreateAnimatedColor(_dangerFlash * 50 / 255f, DangerColor);
                using (var overlayBrush = new SolidBrush(overlayColor))
                {
                    args.Graphics.FillRectangle(overlayBrush, rect);
                }

                // Danger message
                const string dangerText = "⚠️ DAILY LOSS LIMIT REACHED! ⚠️";
                var textSize = args.Graphics.MeasureString(dangerText, _warningFont);
                var textX = rect.X + (rect.Width - textSize.Width) / 2;
                var textY = rect.Y + rect.Height / 2 - 50;

                using (var textBrush = new SolidBrush(CreateAnimatedColor(_dangerFlash, Color.White)))
                {
                    args.Graphics.DrawString(dangerText, _warningFont, textBrush, textX, textY);
                }
            }
            else if (_riskState == 1) // Warning
            {
                // Warning border
                var borderColor = CreateAnimatedColor(_warningPulse * 100 / 255f, WarningColor);
                using (var borderPen = new Pen(borderColor, 5))
                {
                    args.Graphics.DrawRectangle(borderPen, rect.X + 2, rect.Y + 2, rect.Width - 4, rect.Height - 4);
                }
            }
        }

        #region Helper Methods

        /// <summary>
        /// Dispose graphics resources
        /// </summary>
        private void DisposeGraphicsResources()
        {
            _headerFont?.Dispose();
            _sessionFont?.Dispose();
            _warningFont?.Dispose();
            _statsFont?.Dispose();
            _cachedSafeBrush?.Dispose();
            _cachedWarningBrush?.Dispose();
            _cachedDangerBrush?.Dispose();
            _cachedWhiteBrush?.Dispose();
            _cachedGrayBrush?.Dispose();
        }

        /// <summary>
        /// Cleanup resources
        /// </summary>
        protected override void OnClear()
        {
            DisposeGraphicsResources();
            base.OnClear();
        }

        #endregion

        #region Public Properties - Optimized Access

        public double CurrentSessionPnL => _currentSessionPnL;
        public double MaxDrawdown => _maxDrawdown;
        public double MaxProfit => _maxProfit;
        public bool IsWarningActive => _riskState == 1;
        public bool IsDangerActive => _riskState == 2;
        public string CurrentActiveSession => _sessionNames[_currentActiveSessionId];
        public double MFE => _mfe;
        public double MAE => _mae;

        #endregion


        #region Helper Methods - CLEAN GRAPHICS

        /// <summary>
        /// CLEAN: Draw rounded rectangle fill - no extensions needed
        /// </summary>
        private void DrawRoundedRectangle(Graphics graphics, Brush brush, RectangleF rect, float radius)
        {
            using (var path = CreateRoundedRectPath(rect, radius))
            {
                graphics.FillPath(brush, path);
            }
        }

        /// <summary>
        /// CLEAN: Draw rounded rectangle border - no extensions needed  
        /// </summary>
        private void DrawRoundedRectangleBorder(Graphics graphics, Pen pen, RectangleF rect, float radius)
        {
            using (var path = CreateRoundedRectPath(rect, radius))
            {
                graphics.DrawPath(pen, path);
            }
        }

        /// <summary>
        /// CLEAN: Create rounded rectangle path
        /// </summary>
        private GraphicsPath CreateRoundedRectPath(RectangleF rect, float radius)
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

        #endregion
    }
}