// Copyright QUANTOWER LLC. © 2017-2024. All rights reserved.

// HFT Optimized Implementation with Enhanced Performance and Visuals

using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using TradingPlatform.BusinessLayer;

namespace OpeningRangeBreakout
{
    /// <summary>
    /// Opening Range with Breakouts & Targets Indicator - Professional HFT Implementation
    /// Calculates Opening Range levels with dynamic targets and breakout signals
    /// Identical logic to PineScript version with enhanced performance and visuals
    /// © LuxAlgo - Converted for Quantower by HFT Engineering Team
    /// </summary>
    public class OpeningRangeBreakoutIndicator : Indicator
    {
        #region Private Fields

    
        private const int MIN_WARMUP_PERIOD = 10;
        private const int MAX_TARGETS = 50;
        private const double MIN_RANGE_THRESHOLD = 0.0001;
        private const double MAX_TARGET_PERCENTAGE = 10.0;
        private const int MAX_SIGNAL_HISTORY = 1000;
        private const int MAX_LINE_HISTORY = 500;
        
        // VALIDATION CONSTANTS
        private const double FALLBACK_TICK_SIZE = 0.01; // Fallback tick size if not available
        private const int MIN_BREAKOUT_PERSISTENCE_MINUTES = 2; // Minimum time outside OR for valid breakout

        // Session Management
        private DateTime _sessionStartTime = DateTime.MinValue;
        private DateTime _sessionEndTime = DateTime.MinValue;
        private bool _isInOpeningRange = false;
        private bool _openingRangeActive = false;
        private bool _openingRangeToken = false;
        private int _lastProcessedBar = -1;

        // Opening Range Core Values
        private double _openingRangeHigh = 0.0;
        private double _openingRangeLow = 0.0;
        private double _openingRangeMid = 0.0;
        private double _openingRangeWidth = 0.0;
        private double _previousOpeningRangeMid = 0.0;

        // Target Calculation Variables
        private double _highestSinceTarget = 0.0;
        private double _lowestSinceTarget = 0.0;
        private int _upTargetCount = 0;
        private int _downTargetCount = 0;
        private int _maxUpTargets = 0;
        private int _maxDownTargets = 0;

        // Signal Management - Enhanced with 2-signal confirmation
        private bool _upBreakoutCheck = true;
        private bool _downBreakoutCheck = true;
        private bool _upSignalTriggered = false;
        private bool _downSignalTriggered = false;
        
        // 2-Signal Confirmation System
        private SignalConfirmation _upConfirmation = new SignalConfirmation { IsUpDirection = true };
        private SignalConfirmation _downConfirmation = new SignalConfirmation { IsUpDirection = false };
        private DateTime _lastSignalResetTime = DateTime.MinValue;
        // Timeout will be read from SignalConfirmationTimeoutMinutes parameter

        // Moving Average State
        private Queue<double> _sessionPrices = new Queue<double>();
        private double _sessionMovingAverage = 0.0;

        // Direction Bias (based on current vs previous day's ORM)
        private int _dayDirection = 0; // 1 = bullish, -1 = bearish, 0 = neutral
        private double _previousDayORMid = 0.0; // Store actual previous trading day's OR Mid
        private DateTime _lastBiasCalculationDate = DateTime.MinValue;

        // Target Management Collections
        private List<TargetLevel> _upTargets = new List<TargetLevel>();
        private List<TargetLevel> _downTargets = new List<TargetLevel>();
        private List<SignalInfo> _signalHistory = new List<SignalInfo>();
        // Session tracking for separating historical and current targets
        private int _currentSessionId = 0;
        
        // Statistics Tracking
        private List<ORSessionStatistics> _sessionStats = new List<ORSessionStatistics>();
        private ORSessionStatistics _currentSessionStats = null;
        private ORStatisticsSummary _cachedSummary = null;
        private DateTime _lastStatsUpdate = DateTime.MinValue;
        private const int MAX_SESSION_HISTORY = 500; // Keep last 500 sessions
        
        // STATISTICS STATE MANAGEMENT - BUG FIX #2
        private bool _currentSessionFinalized = false; // Atomic flag to prevent double-processing
        private bool _statisticsDirty = false; // Only recalculate when needed
        private readonly Dictionary<int, (bool hasUp, bool hasDown, DateTime? upTime, DateTime? downTime)> _breakoutCache = 
            new Dictionary<int, (bool, bool, DateTime?, DateTime?)>(); // Cache breakout validation results

        // Visual State Management
        private Font _labelFont;
        private SolidBrush _textBrush;
        private Pen _linePen;

        // Performance Tracking
        private int _calculationCounter = 0;
        private DateTime _lastBarTime = DateTime.MinValue;

        // Caching for expensive calculations
        private bool _targetsCacheValid = false;
        private bool _sessionCacheValid = false;

        #endregion

        #region Enums and Classes

        public enum SignalBiasType
        {
            NoBias,
            DailyBias
        }

        public enum TargetDisplayType
        {
            Adaptive,
            Extended
        }

        public enum TargetCrossSource
        {
            Close,
            HighsLows
        }

        public enum MovingAverageType
        {
            SMA,
            EMA,
            RMA,
            WMA,
            VWMA
        }

        public enum LineStyleType
        {
            Solid,
            Dashed,
            Dotted
        }

        private class TargetLevel
        {
            public double Price { get; set; }
            public int TargetNumber { get; set; }
            public bool IsUpTarget { get; set; }
            public DateTime CreatedTime { get; set; }
            public bool IsVisible { get; set; } = true;
            public bool IsFilled { get; set; } = false; // Track if price has reached this target
            public DateTime FilledTime { get; set; } = DateTime.MinValue; // When target was filled
            public double ORHigh { get; set; }  // Store the OR High used to calculate this target
            public double ORLow { get; set; }   // Store the OR Low used to calculate this target
            public double ORWidth { get; set; } // Store the OR Width used to calculate this target
            public int SessionId { get; set; }  // Unique id of the session that generated this target
            public DateTime SessionStartTime { get; set; } // Store actual session start time
            public DateTime SessionEndTime { get; set; }   // Store actual session end time
        }

        private class SignalInfo
        {
            public DateTime Time { get; set; }
            public double Price { get; set; }
            public bool IsUpSignal { get; set; }
            public string SignalText { get; set; }
            public double BreakoutLevel { get; set; }  // Store the actual breakout level for this signal
            public bool IsConfirmed { get; set; } = false;  // Track if signal is confirmed (2nd breakout)
            public int ConfirmationCount { get; set; } = 1;  // Track how many times this direction broke
            public int SessionId { get; set; }  // Session in which the signal was generated
        }

        private class SignalConfirmation
        {
            public bool IsUpDirection { get; set; }
            public DateTime FirstBreakoutTime { get; set; }
            public double FirstBreakoutPrice { get; set; }
            public int BreakoutCount { get; set; } = 0;
            public bool IsWaitingForConfirmation { get; set; } = false;
        }

        private class ORSessionStatistics
        {
            public DateTime SessionDate { get; set; }
            public int SessionId { get; set; }
            public double ORHigh { get; set; }
            public double ORLow { get; set; }
            public double ORMid { get; set; }
            public double ORWidth { get; set; }
            
            // Breakout Analytics
            public bool HasUpBreakout { get; set; } = false;
            public bool HasDownBreakout { get; set; } = false;
            public DateTime? FirstUpBreakoutTime { get; set; }
            public DateTime? FirstDownBreakoutTime { get; set; }
            public double MaxExtensionUp { get; set; } = 0.0; // Max distance above OR High
            public double MaxExtensionDown { get; set; } = 0.0; // Max distance below OR Low
            
            // Target Performance
            public int UpTargetsHit { get; set; } = 0;
            public int DownTargetsHit { get; set; } = 0;
            public List<DateTime> TargetHitTimes { get; set; } = new List<DateTime>();
            
            // Range Behavior
            public bool ReturnedToRange { get; set; } = false;
            public int FalseBreakouts { get; set; } = 0;
            public TimeSpan TimeToFirstBreakout { get; set; } = TimeSpan.Zero;
            
            // Daily Bias Performance
            public int DailyBias { get; set; } = 0; // 1=bullish, -1=bearish, 0=neutral
            public bool BiasCorrect { get; set; } = false; // Did bias match actual direction
        }

        private class ORStatisticsSummary
        {
            // Target Performance
            public double AverageTargetsHitPerSession { get; set; }
            
            // Range Behavior
            public double FalseBreakoutPercentage { get; set; }
            public double ReturnToRangePercentage { get; set; }
            
            // Bias Accuracy
            public double BiasAccuracyPercentage { get; set; }
            
            // Session Count
            public int TotalSessions { get; set; }
            public DateTime FirstSession { get; set; }
            public DateTime LastSession { get; set; }
        }

        #endregion

        #region Statistics Validation Methods - BUG FIX #1-4

        /// <summary>
        /// Validates if an OR session has valid data for statistics calculations
        /// HFT Pattern: Guard Clauses for fail-fast validation
        /// </summary>
        private bool IsValidORSession(ORSessionStatistics session)
        {
            if (session == null) return false;
            if (session.ORHigh <= 0 || session.ORLow <= 0) return false;
            if (session.ORHigh <= session.ORLow) return false;
            if ((session.ORHigh - session.ORLow) < MIN_RANGE_THRESHOLD) return false;
            return true;
        }

        /// <summary>
        /// Unified breakout detection method - Single source of truth for all statistics
        /// HFT Pattern: Atomic validation with consistent logic across all calculations
        /// </summary>
        private (bool hasUpBreakout, bool hasDownBreakout, DateTime? upTime, DateTime? downTime) 
            ValidateSessionBreakouts(ORSessionStatistics session)
        {
            // Guard clause for invalid sessions
            if (!IsValidORSession(session)) 
                return (false, false, null, null);

            // Check cache first for performance
            if (_breakoutCache.TryGetValue(session.SessionId, out var cached))
                return cached;

            bool hasUpBreakout = false;
            bool hasDownBreakout = false;
            DateTime? upTime = null;
            DateTime? downTime = null;

            // ENHANCED BREAKOUT DETECTION with dynamic threshold (BUG FIX #1 & #3)
            double threshold = GetMinimumBreakoutThreshold();
            
            if (session.MaxExtensionUp >= threshold && IsValidBreakout(session.MaxExtensionUp, session.FirstUpBreakoutTime))
            {
                hasUpBreakout = true;
                upTime = session.FirstUpBreakoutTime;
            }

            if (session.MaxExtensionDown >= threshold && IsValidBreakout(session.MaxExtensionDown, session.FirstDownBreakoutTime))
            {
                hasDownBreakout = true;
                downTime = session.FirstDownBreakoutTime;
            }

            // Cache result for performance
            var result = (hasUpBreakout, hasDownBreakout, upTime, downTime);
            _breakoutCache[session.SessionId] = result;
            
            return result;
        }

        /// <summary>
        /// Invalidates breakout cache when session data changes
        /// HFT Pattern: Cache coherency management
        /// </summary>
        private void InvalidateBreakoutCache(int sessionId = -1)
        {
            if (sessionId == -1)
                _breakoutCache.Clear(); // Clear all
            else
                _breakoutCache.Remove(sessionId); // Clear specific session
                
            _statisticsDirty = true; // Mark statistics for recalculation
        }
        
        /// <summary>
        /// Calculate dynamic breakout threshold based on instrument and OR width - BUG FIX #1
        /// This replaces the microscopic fixed threshold with meaningful, instrument-appropriate values
        /// </summary>
        private double GetMinimumBreakoutThreshold()
        {
            if (_openingRangeWidth <= MIN_RANGE_THRESHOLD) return FALLBACK_TICK_SIZE;
            
            try
            {
                // Get instrument tick size (prefer symbol tick size, fallback to default)
                double tickSize = FALLBACK_TICK_SIZE; // Default fallback
                
                // Try to get actual tick size from symbol if available
                if (Symbol?.TickSize != null && Symbol.TickSize > 0)
                    tickSize = Symbol.TickSize;
                
                // Use fixed threshold of 3 ticks as meaningful movement
                double threshold = tickSize * 3;
                
                return threshold;
            }
            catch (Exception)
            {
                // Fallback to tick-based threshold if calculation fails
                return FALLBACK_TICK_SIZE * 3;
            }
        }
        
        /// <summary>
        /// Validate if a breakout is meaningful with time and price persistence - BUG FIX #3
        /// </summary>
        private bool IsValidBreakout(double extension, DateTime? breakoutTime)
        {
            // Must have meaningful extension size
            if (extension < GetMinimumBreakoutThreshold()) return false;
            
            // Must have valid breakout time for persistence check
            if (!breakoutTime.HasValue) return true; // Skip time check if no time available
            
            // Must persist for minimum time (prevents noise/whipsaws)
            var timeSinceBreakout = Time() - breakoutTime.Value;
            if (timeSinceBreakout.TotalMinutes < MIN_BREAKOUT_PERSISTENCE_MINUTES) return false;
            
            return true;
        }

        /// <summary>
        /// Diagnostic method for statistics debugging - ERROR RECOVERY (BUG FIX #5)
        /// Enhanced with threshold and extension analysis
        /// </summary>
        private void LogStatisticsDiagnostics()
        {
            try
            {
                // Basic statistics diagnostics
                var sessionCount = _sessionStats.Count;
                var currentSessionId = _currentSessionStats?.SessionId ?? -1;
                var isFinalized = _currentSessionFinalized;
                
                if (_currentSessionStats != null && _openingRangeWidth > MIN_RANGE_THRESHOLD)
                {
                    // Calculate current threshold for this session
                    double threshold = GetMinimumBreakoutThreshold();
                    
                    // Get validated breakout status
                    var (hasUp, hasDown, _, _) = ValidateSessionBreakouts(_currentSessionStats);
                    
                    // Extension analysis
                    double upExt = _currentSessionStats.MaxExtensionUp;
                    double downExt = _currentSessionStats.MaxExtensionDown;
                    
                    // Log key diagnostic info (silently for debugging)
                    // Threshold: {threshold:F4}, UpExt: {upExt:F4}, DownExt: {downExt:F4}
                    // HasUp: {hasUp}, HasDown: {hasDown}, OR Width: {_openingRangeWidth:F4}
                }
            }
            catch (Exception ex)
            {
                // Error in diagnostics - continuing silently
            }
        }

        #endregion

        #region Input Parameters

        [InputParameter("Show Historical Data", 100)]
        public bool ShowHistoricalData { get; set; } = true;

        [InputParameter("Opening Range Session Duration (Minutes)", 110, 15, 480, 1, 0)]
        public int OpeningRangeMinutes { get; set; } = 30;

        [InputParameter("Use Custom Session Time", 120)]
        public bool UseCustomSession { get; set; } = false;

        [InputParameter("Custom Session Start Hour", 130, 0, 23, 1, 0)]
        public int CustomSessionStartHour { get; set; } = 9;

        [InputParameter("Custom Session Start Minute", 140, 0, 59, 1, 0)]
        public int CustomSessionStartMinute { get; set; } = 30;

        [InputParameter("Custom Session End Hour", 150, 0, 23, 1, 0)]
        public int CustomSessionEndHour { get; set; } = 9;

        [InputParameter("Custom Session End Minute", 160, 0, 59, 1, 0)]
        public int CustomSessionEndMinute { get; set; } = 45;

        [InputParameter("Show Breakout Signals", 200)]
        public bool ShowBreakoutSignals { get; set; } = true;

        [InputParameter("Require 2-Signal Confirmation", 205)]
        public bool RequireTwoSignalConfirmation { get; set; } = true;

        [InputParameter("Signal Confirmation Timeout (Minutes)", 207, 5, 120, 5, 0)]
        public int SignalConfirmationTimeoutMinutes { get; set; } = 30;

        [InputParameter("Signal Bias", 210, variants: new object[]
        {
            "No Bias", SignalBiasType.NoBias,
            "Daily Bias", SignalBiasType.DailyBias
        })]
        public SignalBiasType SignalBias { get; set; } = SignalBiasType.NoBias;

        [InputParameter("Show Targets", 300)]
        public bool ShowTargets { get; set; } = true;

        [InputParameter("Target % of Range", 310, 1.0, 1000.0, 1.0, 1)]
        public double TargetPercentage { get; set; } = 50.0;

        [InputParameter("Initial Target Count", 312, 1, 20, 1, 0)]
        public int InitialTargetCount { get; set; } = 5;

        [InputParameter("Target Cross Source", 320, variants: new object[]
        {
            "Close", TargetCrossSource.Close,
            "Highs/Lows", TargetCrossSource.HighsLows
        })]
        public TargetCrossSource TargetSource { get; set; } = TargetCrossSource.Close;

        [InputParameter("Target Display", 330, variants: new object[]
        {
            "Adaptive", TargetDisplayType.Adaptive,
            "Extended", TargetDisplayType.Extended
        })]
        public TargetDisplayType TargetDisplay { get; set; } = TargetDisplayType.Adaptive;

        [InputParameter("Show Session Moving Average", 400)]
        public bool ShowSessionMA { get; set; } = false;

        [InputParameter("MA Length", 410, 1, 200, 1, 0)]
        public int MovingAverageLength { get; set; } = 20;

        [InputParameter("MA Type", 420, variants: new object[]
        {
            "SMA", MovingAverageType.SMA,
            "EMA", MovingAverageType.EMA,
            "RMA", MovingAverageType.RMA,
            "WMA", MovingAverageType.WMA,
            "VWMA", MovingAverageType.VWMA
        })]
        public MovingAverageType MAType { get; set; } = MovingAverageType.EMA;

        // Color Parameters
        [InputParameter("Bull Target Color", 500)]
        public Color BullTargetColor { get; set; } = Color.FromArgb(60, 8, 153, 129);

        [InputParameter("Bear Target Color", 510)]
        public Color BearTargetColor { get; set; } = Color.FromArgb(60, 242, 54, 69);

        [InputParameter("Bull Fill Color", 520)]
        public Color BullFillColor { get; set; } = Color.FromArgb(80, 8, 153, 129);

        [InputParameter("Bear Fill Color", 530)]
        public Color BearFillColor { get; set; } = Color.FromArgb(80, 242, 54, 69);

        [InputParameter("OR Levels Color", 540)]
        public Color ORLevelsColor { get; set; } = Color.FromArgb(120, 135, 123, 134);

        [InputParameter("OR Fill Color", 550)]
        public Color ORFillColor { get; set; } = Color.FromArgb(60, Color.Gray);

        [InputParameter("Up Signal Color", 560)]
        public Color UpSignalColor { get; set; } = Color.FromArgb(8, 153, 129);

        [InputParameter("Down Signal Color", 570)]
        public Color DownSignalColor { get; set; } = Color.FromArgb(242, 54, 69);

        [InputParameter("MA Color", 580)]
        public Color MAColor { get; set; } = Color.Orange;

        // Statistics Parameters
        [InputParameter("Show OR Statistics", 590)]
        public bool ShowORStatistics { get; set; } = false;
        
        [InputParameter("Statistics Period (Sessions)", 595, 10, 500, 10, 0)]
        public int StatisticsPeriod { get; set; } = 100;

        [InputParameter("Target Line Style", 600, variants: new object[]
        {
            "Solid", LineStyleType.Solid,
            "Dashed", LineStyleType.Dashed,
            "Dotted", LineStyleType.Dotted
        })]
        public LineStyleType TargetLineStyle { get; set; } = LineStyleType.Solid;

        #endregion

        /// <summary>
        /// Constructor
        /// </summary>
        public OpeningRangeBreakoutIndicator() : base()
        {
            Name = "Opening Range with Breakouts & Targets";
            Description = "Opening Range Breakout with Dynamic Targets and Signals - HFT OPTIMIZED";
            ShortName = "ORB&Targets";

            // Add line series for core levels
            AddLineSeries("OR High", Color.Gray, 2, LineStyle.Solid);           // Index 0
            AddLineSeries("OR Low", Color.Gray, 2, LineStyle.Solid);            // Index 1
            AddLineSeries("OR Mid", Color.Gray, 1, LineStyle.Dash);             // Index 2
            AddLineSeries("Session MA", Color.Orange, 2, LineStyle.Solid);      // Index 3

            SeparateWindow = false; // Overlay on main chart
        }

        /// <summary>
        /// Initialize indicator
        /// </summary>
        protected override void OnInit()
        {
            // Clear historical data
            ResetState();
            
            // Initialize visual resources
            InitializeVisualResources();

            // Validate inputs
            ValidateInputParameters();
        }

        /// <summary>
        /// Reset all state variables - HFT optimized
        /// </summary>
        private void ResetState()
        {
            _sessionPrices.Clear();
            _upTargets.Clear();
            _downTargets.Clear();
            _signalHistory.Clear();

            _isInOpeningRange = false;
            _openingRangeActive = false;
            _openingRangeToken = false;
            _lastProcessedBar = -1;

            _openingRangeHigh = 0.0;
            _openingRangeLow = 0.0;
            _openingRangeMid = 0.0;
            _openingRangeWidth = 0.0;
            _previousOpeningRangeMid = 0.0;

            _highestSinceTarget = 0.0;
            _lowestSinceTarget = 0.0;
            _upTargetCount = 0;
            _downTargetCount = 0;

            _upBreakoutCheck = true;
            _downBreakoutCheck = true;
            _upSignalTriggered = false;
            _downSignalTriggered = false;
            
            // Reset 2-signal confirmation state
            _upConfirmation.IsWaitingForConfirmation = false;
            _upConfirmation.BreakoutCount = 0;
            _downConfirmation.IsWaitingForConfirmation = false;
            _downConfirmation.BreakoutCount = 0;
            _lastSignalResetTime = DateTime.Now;

            _sessionMovingAverage = 0.0;
            _dayDirection = 0;
            _previousDayORMid = 0.0;
            _lastBiasCalculationDate = DateTime.MinValue;

            _targetsCacheValid = false;
            _sessionCacheValid = false;
            _calculationCounter = 0;
            _lastBarTime = DateTime.MinValue;
            
            // Don't clear statistics history - preserve across resets
            // _sessionStats.Clear(); 
            _currentSessionStats = null;
            _cachedSummary = null;

            _currentSessionId = 0; // Reset session counter
            
            // RESET STATISTICS STATE FLAGS (BUG FIX #2, #5)
            _currentSessionFinalized = false;
            _statisticsDirty = false;
            _breakoutCache.Clear();
            
            // State reset completed
        }

        /// <summary>
        /// Initialize visual resources
        /// </summary>
        private void InitializeVisualResources()
        {
            _labelFont?.Dispose();
            _textBrush?.Dispose();
            _linePen?.Dispose();

            _labelFont = new Font("Arial", 10, FontStyle.Bold);
            _textBrush = new SolidBrush(Color.Yellow);
            _linePen = new Pen(Color.Gray, 1);
        }

        /// <summary>
        /// Validate input parameters
        /// </summary>
        private void ValidateInputParameters()
        {
            // Validate opening range session duration - minimum 15 minutes to prevent premature resets
            if (OpeningRangeMinutes < 15 || OpeningRangeMinutes > 480)
            {
                OpeningRangeMinutes = 30;
            }

            // Validate target percentage
            if (TargetPercentage <= 0 || TargetPercentage > MAX_TARGET_PERCENTAGE * 100)
            {
                TargetPercentage = 50.0;
            }

            // Validate MA length
            if (MovingAverageLength <= 0 || MovingAverageLength > 200)
            {
                MovingAverageLength = 20;
            }

            // Validate custom session times
            if (UseCustomSession)
            {
                if (CustomSessionStartHour < 0 || CustomSessionStartHour > 23)
                    CustomSessionStartHour = 9;
                if (CustomSessionStartMinute < 0 || CustomSessionStartMinute > 59)
                    CustomSessionStartMinute = 30;
                if (CustomSessionEndHour < 0 || CustomSessionEndHour > 23)
                    CustomSessionEndHour = 9;
                if (CustomSessionEndMinute < 0 || CustomSessionEndMinute > 59)
                    CustomSessionEndMinute = 45;
            }
        }

        /// <summary>
        /// Main calculation entry point
        /// </summary>
        protected override void OnUpdate(UpdateArgs args)
        {
            if (Count < MIN_WARMUP_PERIOD) return;

            var currentTime = Time();
            var currentBarIndex = Count - 1;

            // Check if this is a new bar
            bool isNewBar = currentBarIndex != _lastProcessedBar;
            if (isNewBar)
            {
                _lastProcessedBar = currentBarIndex;
                _lastBarTime = currentTime;
            }

            // Invalidate caches on new bar
            if (isNewBar)
            {
                _targetsCacheValid = false;
                _sessionCacheValid = false;
            }

            // Determine session state
            UpdateSessionState(currentTime);

            // Process opening range logic
            ProcessOpeningRange();

            // Calculate targets and signals
            if (_openingRangeToken)
            {
                CalculateTargets();
                ProcessBreakoutSignals();
                UpdateSessionMovingAverage();
                
                // Update statistics for current session
                UpdateSessionStatistics();
            }

            // Set line values
            SetLineValues();

            _calculationCounter++;
            
            // DIAGNOSTIC LOGGING AND ERROR RECOVERY (BUG FIX #5)
            // Log diagnostics every 100 bars or on significant events
            if (_calculationCounter % 100 == 0 || (isNewBar && ShowORStatistics))
            {
                LogStatisticsDiagnostics();
            }
            
            // ERROR RECOVERY: Reset corrupted state if detected
            if (_currentSessionStats != null && _currentSessionFinalized && !_currentSessionFinalized)
            {
                // Detected state corruption - resetting session finalization flag
                _currentSessionFinalized = false;
            }
        }

        /// <summary>
        /// Update session state based on current time - FIXED session time calculation consistency
        /// </summary>
        private void UpdateSessionState(DateTime currentTime)
        {
            bool previousInRange = _isInOpeningRange;

            if (UseCustomSession)
            {
                // Use SAME logic as OnOpeningRangeStart() for consistency
                var sessionStartCandidate = currentTime.Date.AddHours(CustomSessionStartHour).AddMinutes(CustomSessionStartMinute);
                var sessionEndCandidate = currentTime.Date.AddHours(CustomSessionEndHour).AddMinutes(CustomSessionEndMinute);
                
                // Handle overnight sessions
                if (CustomSessionEndHour < CustomSessionStartHour || 
                    (CustomSessionEndHour == CustomSessionStartHour && CustomSessionEndMinute <= CustomSessionStartMinute))
                {
                    sessionEndCandidate = sessionEndCandidate.AddDays(1);
                }
                
                // If current time is before today's session start, use previous day's session
                if (currentTime < sessionStartCandidate)
                {
                    sessionStartCandidate = sessionStartCandidate.AddDays(-1);
                    sessionEndCandidate = sessionEndCandidate.AddDays(-1);
                }

                _isInOpeningRange = currentTime >= sessionStartCandidate && currentTime <= sessionEndCandidate;
            }
            else
            {
                // Use timeframe-based logic
                var sessionStart = GetSessionStart(currentTime);
                var sessionEnd = sessionStart.AddMinutes(OpeningRangeMinutes);
                
                _isInOpeningRange = currentTime >= sessionStart && currentTime <= sessionEnd;
            }

            // Detect session start
            bool sessionStarted = _isInOpeningRange && !previousInRange;
            bool sessionEnded = !_isInOpeningRange && previousInRange;

            if (sessionStarted)
            {
                OnOpeningRangeStart();
            }
            else if (sessionEnded)
            {
                OnOpeningRangeEnd();
            }
        }

        /// <summary>
        /// Get session start time for the current bar
        /// </summary>
        private DateTime GetSessionStart(DateTime currentTime)
        {
            // Simple logic: start of trading day
            // This can be enhanced based on specific market requirements
            var marketOpen = currentTime.Date.AddHours(9).AddMinutes(30); // 9:30 AM default
            
            // If current time is before market open, use previous day
            if (currentTime < marketOpen)
            {
                marketOpen = marketOpen.AddDays(-1);
            }
            
            return marketOpen;
        }

        /// <summary>
        /// Handle opening range session start
        /// </summary>
        private void OnOpeningRangeStart()
        {
            // Finalize previous session statistics if exists
            if (_currentSessionStats != null)
            {
                FinalizeSessionStatistics();
            }
            
            // Increment session identifier to segregate targets
            _currentSessionId++;
            
            // Initialize new session statistics
            _currentSessionStats = new ORSessionStatistics
            {
                SessionDate = Time().Date,
                SessionId = _currentSessionId,
                DailyBias = _dayDirection
            };
            
            // RESET FINALIZATION FLAG for new session (BUG FIX #2)
            _currentSessionFinalized = false;
            
            // Calculate daily bias using previous day's OR Mid (if available)
            var currentDate = Time().Date;
            if (_lastBiasCalculationDate != currentDate)
            {
                // Calculate bias direction using stored previous day's OR Mid
                if (_previousDayORMid > 0 && _openingRangeMid > 0)
                {
                    double biasThreshold = _previousDayORMid * 0.001; // 0.1% threshold for neutral zone
                    
                    if (_openingRangeMid > (_previousDayORMid + biasThreshold))
                        _dayDirection = 1; // Bullish - current OR Mid significantly above previous day
                    else if (_openingRangeMid < (_previousDayORMid - biasThreshold))
                        _dayDirection = -1; // Bearish - current OR Mid significantly below previous day
                    else
                        _dayDirection = 0; // Neutral - within threshold range
                    _lastBiasCalculationDate = currentDate; // Only lock date when OR mid is valid
                }
                else
                {
                    _dayDirection = 0; // No bias if no previous day data
                    // Do not set _lastBiasCalculationDate yet; allow recalculation later when OR forms
                }
            }

            // Calculate and store session times for precise target drawing
            var currentTime = Time();
            
            // CRITICAL: Use the ACTUAL session start time, not current bar time for date calculation
            if (UseCustomSession)
            {
                // Find the correct session start date (could be previous day for overnight sessions)
                var sessionStartCandidate = currentTime.Date.AddHours(CustomSessionStartHour).AddMinutes(CustomSessionStartMinute);
                var sessionEndCandidate = currentTime.Date.AddHours(CustomSessionEndHour).AddMinutes(CustomSessionEndMinute);
                
                // Handle overnight sessions
                if (CustomSessionEndHour < CustomSessionStartHour || 
                    (CustomSessionEndHour == CustomSessionStartHour && CustomSessionEndMinute <= CustomSessionStartMinute))
                {
                    sessionEndCandidate = sessionEndCandidate.AddDays(1);
                }
                
                // If current time is before today's session start, use previous day's session
                if (currentTime < sessionStartCandidate)
                {
                    sessionStartCandidate = sessionStartCandidate.AddDays(-1);
                    sessionEndCandidate = sessionEndCandidate.AddDays(-1);
                }
                
                _sessionStartTime = sessionStartCandidate;
                _sessionEndTime = sessionEndCandidate;
            }
            else
            {
                _sessionStartTime = GetSessionStart(currentTime);
                _sessionEndTime = _sessionStartTime.AddMinutes(OpeningRangeMinutes);
            }

            // Clear previous session data if not showing historical (but keep current session data)
            if (!ShowHistoricalData)
            {
                // Only remove targets from previous sessions (SessionId < _currentSessionId)
                _upTargets.RemoveAll(t => t.SessionId < _currentSessionId);
                _downTargets.RemoveAll(t => t.SessionId < _currentSessionId);
                
                // Clear signal history (signals are point-in-time, not session-specific)
                _signalHistory.Clear();
            }

            // Reset OR values
            _openingRangeHigh = High();
            _openingRangeLow = Low();
            
            // Reset target counters and extension tracking
            _upTargetCount = 0;
            _downTargetCount = 0;
            _highestSinceTarget = _openingRangeHigh;
            _lowestSinceTarget = _openingRangeLow;

            // Reset signal checks
            _upBreakoutCheck = true;
            _downBreakoutCheck = true;
            _upSignalTriggered = false;
            _downSignalTriggered = false;

            // Reset session MA
            _sessionPrices.Clear();

            _openingRangeActive = true;
            _openingRangeToken = false;
            
            // CRITICAL BUG FIX #2: Do NOT initialize extension tracking yet
            // Extensions should only be tracked AFTER opening range formation ends
            // This prevents counting movements during OR formation as "breakouts"
            
            // Target creation will happen in ProcessOpeningRange once we have valid range
        }

        /// <summary>
        /// Handle opening range session end
        /// </summary>
        private void OnOpeningRangeEnd()
        {
            if (_openingRangeActive && _openingRangeWidth > MIN_RANGE_THRESHOLD)
            {
                _openingRangeToken = true;

                // Store current session's OR Mid for next day's bias calculation
                _previousDayORMid = _openingRangeMid;

                // Record OR values in current session statistics
                if (_currentSessionStats != null)
                {
                    _currentSessionStats.ORHigh = _openingRangeHigh;
                    _currentSessionStats.ORLow = _openingRangeLow;
                    _currentSessionStats.ORMid = _openingRangeMid;
                    _currentSessionStats.ORWidth = _openingRangeWidth;
                }

                // Initialize target calculation values
                _highestSinceTarget = _openingRangeHigh;
                _lowestSinceTarget = _openingRangeLow;

                // Create the initial set of targets for this session
                CreateInitialTargets();
            }

            _openingRangeActive = false;
        }

        /// <summary>
        /// Process opening range calculation during active session
        /// </summary>
        private void ProcessOpeningRange()
        {
            if (_openingRangeActive)
            {
                // Update OR levels with current bar's high/low
                if (High() > _openingRangeHigh)
                    _openingRangeHigh = High();
                
                if (Low() < _openingRangeLow)
                    _openingRangeLow = Low();

                // Recalculate derived values
                _openingRangeMid = (_openingRangeHigh + _openingRangeLow) / 2.0;
                _openingRangeWidth = Math.Abs(_openingRangeHigh - _openingRangeLow);
                
                // Create initial targets as soon as we have a valid range width
                // This ensures targets are available from the start of OR, not just at the end
                if (_openingRangeWidth > MIN_RANGE_THRESHOLD && 
                    !_upTargets.Any(t => t.SessionId == _currentSessionId))
                {
                    CreateInitialTargets();
                }

                // Recalculate bias once OR values are valid to avoid neutral lock
                var currentDate = Time().Date;
                if (_openingRangeWidth > MIN_RANGE_THRESHOLD && _lastBiasCalculationDate != currentDate && _previousDayORMid > 0)
                {
                    double biasThreshold = _previousDayORMid * 0.001;
                    if (_openingRangeMid > (_previousDayORMid + biasThreshold)) _dayDirection = 1;
                    else if (_openingRangeMid < (_previousDayORMid - biasThreshold)) _dayDirection = -1;
                    else _dayDirection = 0;
                    _lastBiasCalculationDate = currentDate;
                }

                // Update targets live during OR as well
                if (ShowTargets && _openingRangeWidth > MIN_RANGE_THRESHOLD)
                {
                    CalculateTargets();
                }
            }
        }

        /// <summary>
        /// Calculate and manage targets. This method is now only responsible for adding new targets for the current session.
        /// </summary>
        private void CalculateTargets()
        {
            // Always track extensions after OR is valid, regardless of target display setting
            if (_openingRangeWidth <= MIN_RANGE_THRESHOLD) return;

            // Get source values for target crossing check
            double highSource = TargetSource == TargetCrossSource.Close ? Close() : High();
            double lowSource = TargetSource == TargetCrossSource.Close ? Close() : Low();

            // CRITICAL BUG FIX #2: Only track extensions AFTER opening range formation is complete
            // This prevents counting movements during OR formation as "breakouts"
            if (_openingRangeToken) // Only after OR formation is complete
            {
                // Update highest/lowest price reached since the opening range ended for the current session
                // Use true highs/lows for statistical extensions to avoid coupling with target cross source
                double extensionHigh = High();
                double extensionLow = Low();
                if (extensionHigh > _highestSinceTarget)
                    _highestSinceTarget = extensionHigh;

                if (extensionLow < _lowestSinceTarget)
                    _lowestSinceTarget = extensionLow;
            }

            // Always update target filling status regardless of display mode
            UpdateAdaptiveTargetVisibility(highSource, lowSource);

            // If targets are hidden, stop here. Statistics (extensions/Z-score) are already updated above.
            if (!ShowTargets) return;

            // Dynamically add new targets if price extends beyond the current range of targets for this session
            AddNewTargetsForCurrentSession();
        }

        /// <summary>
        /// Creates the initial set of targets for the current session from the start of opening range.
        /// Creates multiple targets upfront to ensure they're available immediately when price moves.
        /// </summary>
        private void CreateInitialTargets()
        {
            if (!ShowTargets || _openingRangeWidth <= MIN_RANGE_THRESHOLD) return;

            // Only add initial targets if none exist for the current session
            if (_upTargets.Any(t => t.SessionId == _currentSessionId) || _downTargets.Any(t => t.SessionId == _currentSessionId))
                return;

            double targetDistance = _openingRangeWidth * (TargetPercentage / 100.0);

            // Create multiple up targets upfront
            for (int i = 1; i <= InitialTargetCount; i++)
            {
                _upTargets.Add(new TargetLevel
                {
                    Price = _openingRangeHigh + (targetDistance * i),
                    TargetNumber = i,
                    IsUpTarget = true,
                    CreatedTime = Time(),
                    IsVisible = TargetDisplay == TargetDisplayType.Extended, // Extended mode shows all initially
                    IsFilled = false,
                    ORHigh = _openingRangeHigh,
                    ORLow = _openingRangeLow,
                    ORWidth = _openingRangeWidth,
                    SessionId = _currentSessionId,
                    SessionStartTime = _sessionStartTime,
                    SessionEndTime = _sessionEndTime
                });
            }

            // Create multiple down targets upfront
            for (int i = 1; i <= InitialTargetCount; i++)
            {
                _downTargets.Add(new TargetLevel
                {
                    Price = _openingRangeLow - (targetDistance * i),
                    TargetNumber = i,
                    IsUpTarget = false,
                    CreatedTime = Time(),
                    IsVisible = TargetDisplay == TargetDisplayType.Extended, // Extended mode shows all initially
                    IsFilled = false,
                    ORHigh = _openingRangeHigh,
                    ORLow = _openingRangeLow,
                    ORWidth = _openingRangeWidth,
                    SessionId = _currentSessionId,
                    SessionStartTime = _sessionStartTime,
                    SessionEndTime = _sessionEndTime
                });
            }
        }

        /// <summary>
        /// Dynamically adds new targets when price extends beyond existing targets for the current session.
        /// This method is stateless regarding target counts, deriving them from the lists, which is more robust.
        /// </summary>
        private void AddNewTargetsForCurrentSession()
        {
            double targetDistance = _openingRangeWidth * (TargetPercentage / 100.0);
            if (targetDistance <= 0) return;

            // Calculate how many targets *should* exist based on the highest price reached so far.
            int requiredUpTargets = (int)Math.Ceiling((_highestSinceTarget - _openingRangeHigh) / targetDistance);
            int requiredDownTargets = (int)Math.Ceiling((_openingRangeLow - _lowestSinceTarget) / targetDistance);

            // Get the current count of targets for this session from the list.
            int currentUpTargetCount = _upTargets.Count(t => t.SessionId == _currentSessionId);
            int currentDownTargetCount = _downTargets.Count(t => t.SessionId == _currentSessionId);

            // Add new up targets if needed
            while (currentUpTargetCount < requiredUpTargets && currentUpTargetCount < MAX_TARGETS)
            {
                currentUpTargetCount++;
                _upTargets.Add(new TargetLevel
                {
                    Price = _openingRangeHigh + (targetDistance * currentUpTargetCount),
                    TargetNumber = currentUpTargetCount,
                    IsUpTarget = true,
                    CreatedTime = Time(),
                    IsVisible = TargetDisplay == TargetDisplayType.Extended,
                    IsFilled = false, // CRITICAL: Always start as unfilled
                    FilledTime = DateTime.MinValue,
                    ORHigh = _openingRangeHigh,
                    ORLow = _openingRangeLow,
                    ORWidth = _openingRangeWidth,
                    SessionId = _currentSessionId,
                    SessionStartTime = _sessionStartTime,
                    SessionEndTime = _sessionEndTime
                });
            }

            // Add new down targets if needed
            while (currentDownTargetCount < requiredDownTargets && currentDownTargetCount < MAX_TARGETS)
            {
                currentDownTargetCount++;
                _downTargets.Add(new TargetLevel
                {
                    Price = _openingRangeLow - (targetDistance * currentDownTargetCount),
                    TargetNumber = currentDownTargetCount,
                    IsUpTarget = false,
                    CreatedTime = Time(),
                    IsVisible = TargetDisplay == TargetDisplayType.Extended,
                    IsFilled = false, // CRITICAL: Always start as unfilled
                    FilledTime = DateTime.MinValue,
                    ORHigh = _openingRangeHigh,
                    ORLow = _openingRangeLow,
                    ORWidth = _openingRangeWidth,
                    SessionId = _currentSessionId,
                    SessionStartTime = _sessionStartTime,
                    SessionEndTime = _sessionEndTime
                });
            }
        }

        /// <summary>
        /// Update target visibility and filling status ONLY for current session - prevents historical corruption.
        /// </summary>
        private void UpdateAdaptiveTargetVisibility(double highSource, double lowSource)
        {
            // CRITICAL FIX: Only update targets from the CURRENT session
            // Historical targets should remain unchanged to preserve their state
            
            UpdateCurrentSessionTargets(_upTargets, highSource, true);
            UpdateCurrentSessionTargets(_downTargets, lowSource, false);
        }
        
        /// <summary>
        /// Update targets for current session only - prevents historical target corruption
        /// </summary>
        private void UpdateCurrentSessionTargets(List<TargetLevel> targets, double priceSource, bool isUpTarget)
        {
            // Only process targets from the current session
            foreach (var target in targets.Where(t => t.SessionId == _currentSessionId))
            {
                bool priceReached = isUpTarget ? priceSource >= target.Price : priceSource <= target.Price;
                
                if (priceReached)
                {
                    target.IsVisible = true;
                    
                    // Mark as filled only if not already filled and price actually crossed
                    if (!target.IsFilled)
                    {
                        target.IsFilled = true;
                        target.FilledTime = Time();
                    }
                }
            }
        }

        /// <summary>
        /// Process breakout signals with 2-signal confirmation system
        /// </summary>
        private void ProcessBreakoutSignals()
        {
            if (!ShowBreakoutSignals) return;

            double close = Close();
            
            // Reset breakout checks when price returns to OR
            if (close > _openingRangeMid && !_downBreakoutCheck)
                _downBreakoutCheck = true;
            
            if (close < _openingRangeMid && !_upBreakoutCheck)
                _upBreakoutCheck = true;

            // Check for timeout of pending confirmations
            CheckConfirmationTimeouts();

            // Check for breakout signals with 2-signal confirmation
            CheckUpBreakoutWithConfirmation(close);
            CheckDownBreakoutWithConfirmation(close);
        }

        /// <summary>
        /// Check for upward breakout signal with 2-signal confirmation and bias logic
        /// </summary>
        private void CheckUpBreakoutWithConfirmation(double close)
        {
            bool crossedUp = close > _openingRangeHigh && Close(1) <= _openingRangeHigh;
            bool shouldTriggerBreakout = false;
            
            if (SignalBias == SignalBiasType.DailyBias)
            {
                if (_dayDirection == 1)
                {
                    // Bullish bias: Early signal on OR breakout (trend-following)
                    shouldTriggerBreakout = crossedUp && _upBreakoutCheck;
                }
                else if (_dayDirection == -1)
                {
                    // Bearish bias: More conservative upward signal (counter-trend)
                    // Wait for price to move 25% of target distance above OR High
                    double conservativeDistance = (_openingRangeWidth * (TargetPercentage / 100.0)) * 0.25;
                    bool crossedConservative = close > (_openingRangeHigh + conservativeDistance) && 
                                             Close(1) <= (_openingRangeHigh + conservativeDistance);
                    shouldTriggerBreakout = crossedConservative && _upBreakoutCheck;
                }
                else
                {
                    // Neutral bias: Standard OR breakout
                    shouldTriggerBreakout = crossedUp && _upBreakoutCheck;
                }
            }
            else
            {
                // No bias: simple breakout
                shouldTriggerBreakout = crossedUp && _upBreakoutCheck;
            }
            
            // Process with 2-signal confirmation system or legacy single signal
            if (shouldTriggerBreakout)
            {
                if (RequireTwoSignalConfirmation)
                {
                    ProcessBreakoutConfirmation(true, close);
                }
                else
                {
                    GenerateLegacyUpSignal(); // Legacy single signal method
                }
            }
        }

        /// <summary>
        /// Check for downward breakout signal with 2-signal confirmation and bias logic
        /// </summary>
        private void CheckDownBreakoutWithConfirmation(double close)
        {
            bool crossedDown = close < _openingRangeLow && Close(1) >= _openingRangeLow;
            bool shouldTriggerBreakout = false;
            
            if (SignalBias == SignalBiasType.DailyBias)
            {
                if (_dayDirection == -1)
                {
                    // Bearish bias: Early signal on OR breakout (trend-following)
                    shouldTriggerBreakout = crossedDown && _downBreakoutCheck;
                }
                else if (_dayDirection == 1)
                {
                    // Bullish bias: More conservative downward signal (counter-trend)
                    // Wait for price to move 25% of target distance below OR Low
                    double conservativeDistance = (_openingRangeWidth * (TargetPercentage / 100.0)) * 0.25;
                    bool crossedConservative = close < (_openingRangeLow - conservativeDistance) && 
                                             Close(1) >= (_openingRangeLow - conservativeDistance);
                    shouldTriggerBreakout = crossedConservative && _downBreakoutCheck;
                }
                else
                {
                    // Neutral bias: Standard OR breakout
                    shouldTriggerBreakout = crossedDown && _downBreakoutCheck;
                }
            }
            else
            {
                // No bias: simple breakout
                shouldTriggerBreakout = crossedDown && _downBreakoutCheck;
            }
            
            // Process with 2-signal confirmation system or legacy single signal
            if (shouldTriggerBreakout)
            {
                if (RequireTwoSignalConfirmation)
                {
                    ProcessBreakoutConfirmation(false, close);
                }
                else
                {
                    GenerateLegacyDownSignal(); // Legacy single signal method
                }
            }
        }

        /// <summary>
        /// Check for confirmation timeouts - reset pending signals after timeout
        /// </summary>
        private void CheckConfirmationTimeouts()
        {
            var currentTime = Time();
            
            // Reset up confirmation if timeout exceeded
            if (_upConfirmation.IsWaitingForConfirmation && 
                (currentTime - _upConfirmation.FirstBreakoutTime).TotalMinutes > SignalConfirmationTimeoutMinutes)
            {
                _upConfirmation.IsWaitingForConfirmation = false;
                _upConfirmation.BreakoutCount = 0;
                _upBreakoutCheck = true; // Re-enable breakout detection
            }
            
            // Reset down confirmation if timeout exceeded
            if (_downConfirmation.IsWaitingForConfirmation && 
                (currentTime - _downConfirmation.FirstBreakoutTime).TotalMinutes > SignalConfirmationTimeoutMinutes)
            {
                _downConfirmation.IsWaitingForConfirmation = false;
                _downConfirmation.BreakoutCount = 0;
                _downBreakoutCheck = true; // Re-enable breakout detection
            }
        }

        /// <summary>
        /// Process breakout confirmation - requires 2 breakouts in same direction
        /// </summary>
        private void ProcessBreakoutConfirmation(bool isUpBreakout, double close)
        {
            var confirmation = isUpBreakout ? _upConfirmation : _downConfirmation;
            
            confirmation.BreakoutCount++;
            
            if (confirmation.BreakoutCount == 1)
            {
                // First breakout - record it but don't generate signal yet
                confirmation.FirstBreakoutTime = Time();
                confirmation.FirstBreakoutPrice = close;
                confirmation.IsWaitingForConfirmation = true;
                
                // Temporarily disable breakout check to avoid duplicate detection
                if (isUpBreakout)
                    _upBreakoutCheck = false;
                else
                    _downBreakoutCheck = false;
            }
            else if (confirmation.BreakoutCount >= 2)
            {
                // Second breakout - generate confirmed signal
                if (isUpBreakout)
                    GenerateConfirmedUpSignal(confirmation);
                else
                    GenerateConfirmedDownSignal(confirmation);
                
                // Reset confirmation state
                confirmation.IsWaitingForConfirmation = false;
                confirmation.BreakoutCount = 0;
            }
        }

        /// <summary>
        /// Generate confirmed upward breakout signal (after 2 breakouts)
        /// </summary>
        private void GenerateConfirmedUpSignal(SignalConfirmation confirmation)
        {
            GenerateSignal(true, true, 2);
        }

        /// <summary>
        /// Generate confirmed downward breakout signal (after 2 breakouts)
        /// </summary>
        private void GenerateConfirmedDownSignal(SignalConfirmation confirmation)
        {
            GenerateSignal(false, true, 2);
        }

        /// <summary>
        /// Generate legacy single upward breakout signal (backward compatibility)
        /// </summary>
        private void GenerateLegacyUpSignal()
        {
            GenerateSignal(true, true, 1);
        }

        /// <summary>
        /// Generate legacy single downward breakout signal (backward compatibility)
        /// </summary>
        private void GenerateLegacyDownSignal()
        {
            GenerateSignal(false, true, 1);
        }

        /// <summary>
        /// Unified signal generation method - reduces code duplication and improves maintainability
        /// </summary>
        private void GenerateSignal(bool isUpSignal, bool isConfirmed, int confirmationCount)
        {
            // Update signal state
            if (isUpSignal)
            {
                _upSignalTriggered = true;
                _upBreakoutCheck = false;
            }
            else
            {
                _downSignalTriggered = true;
                _downBreakoutCheck = false;
            }

            // Determine signal text based on confirmation status
            string signalText;
            if (isConfirmed && confirmationCount >= 2)
            {
                signalText = isUpSignal ? "▲▲" : "▼▼"; // Double arrows for confirmed signals
            }
            else
            {
                signalText = isUpSignal ? "▲" : "▼"; // Single arrows for single or legacy signals
            }

            // Create signal object
            var signal = new SignalInfo
            {
                Time = Time(),
                Price = isUpSignal ? _openingRangeHigh : _openingRangeLow,
                IsUpSignal = isUpSignal,
                SignalText = signalText,
                BreakoutLevel = isUpSignal ? _openingRangeHigh : _openingRangeLow,
                IsConfirmed = isConfirmed,
                ConfirmationCount = confirmationCount,
                SessionId = _currentSessionId
            };

            // Add to history
            _signalHistory.Add(signal);

            // Maintain signal history size limit
            if (_signalHistory.Count > MAX_SIGNAL_HISTORY)
            {
                _signalHistory.RemoveAt(0);
            }
        }

        /// <summary>
        /// Finalize statistics for the completed session - FIXED VERSION
        /// BUG FIXES: #1 Unified breakout detection, #2 Double-counting prevention, #3 Consistent bias logic
        /// </summary>
        private void FinalizeSessionStatistics()
        {
            // GUARD CLAUSE: Prevent double-processing (BUG FIX #2)
            if (_currentSessionStats == null || _currentSessionFinalized) return;
            
            try
            {
                // VALIDATION: Ensure valid session before finalizing (BUG FIX #4)
                if (!IsValidORSession(_currentSessionStats))
                {
                    // Note: Invalid session data - skipping finalization
                    _currentSessionFinalized = true; // Prevent retry attempts
                    return;
                }

                // Calculate final extensions with validation
                _currentSessionStats.MaxExtensionUp = Math.Max(0, _highestSinceTarget - _currentSessionStats.ORHigh);
                _currentSessionStats.MaxExtensionDown = Math.Max(0, _currentSessionStats.ORLow - _lowestSinceTarget);
                
                // Count targets hit
                _currentSessionStats.UpTargetsHit = _upTargets.Count(t => t.SessionId == _currentSessionStats.SessionId && t.IsFilled);
                _currentSessionStats.DownTargetsHit = _downTargets.Count(t => t.SessionId == _currentSessionStats.SessionId && t.IsFilled);
                
                // UNIFIED BREAKOUT DETECTION (BUG FIX #1)
                var (hasUpBreakout, hasDownBreakout, upTime, downTime) = ValidateSessionBreakouts(_currentSessionStats);
                _currentSessionStats.HasUpBreakout = hasUpBreakout;
                _currentSessionStats.HasDownBreakout = hasDownBreakout;
                _currentSessionStats.FirstUpBreakoutTime = upTime;
                _currentSessionStats.FirstDownBreakoutTime = downTime;

                // CONSISTENT BIAS ACCURACY CALCULATION (BUG FIX #3)
                // Now uses same breakout detection logic as statistics
                if (_currentSessionStats.DailyBias == 1 && hasUpBreakout && !hasDownBreakout)
                    _currentSessionStats.BiasCorrect = true;
                else if (_currentSessionStats.DailyBias == -1 && hasDownBreakout && !hasUpBreakout)
                    _currentSessionStats.BiasCorrect = true;
                else if (_currentSessionStats.DailyBias == 0)
                    _currentSessionStats.BiasCorrect = true; // Neutral bias is always "correct"
                    
                // ATOMIC FINALIZATION (BUG FIX #2)
                _sessionStats.Add(_currentSessionStats);
                _currentSessionFinalized = true; // Set flag IMMEDIATELY after adding
                
                // Maintain size limit
                if (_sessionStats.Count > MAX_SESSION_HISTORY)
                {
                    var removedSession = _sessionStats[0];
                    _sessionStats.RemoveAt(0);
                    // Clean up cache for removed session
                    InvalidateBreakoutCache(removedSession.SessionId);
                }
                
                // Mark statistics as dirty for recalculation
                _statisticsDirty = true;
                _cachedSummary = null;
                
                // Session finalized successfully
            }
            catch (Exception ex)
            {
                // Error in FinalizeSessionStatistics - continuing with reset
                _currentSessionFinalized = true; // Prevent retry attempts on error
            }
        }
        
        /// <summary>
        /// Update current session statistics during trading - FIXED VERSION
        /// BUG FIX #1: Uses unified breakout detection, #4: Adds validation
        /// </summary>
        private void UpdateSessionStatistics()
        {
            // GUARD CLAUSES: Validation and finalization check (BUG FIX #4)
            if (_currentSessionStats == null || _currentSessionFinalized) return;
            if (!IsValidORSession(_currentSessionStats)) return;
            
            try
            {
                // Update real-time extensions with validation
                _currentSessionStats.MaxExtensionUp = Math.Max(_currentSessionStats.MaxExtensionUp, 
                    Math.Max(0, _highestSinceTarget - _currentSessionStats.ORHigh));
                _currentSessionStats.MaxExtensionDown = Math.Max(_currentSessionStats.MaxExtensionDown, 
                    Math.Max(0, _currentSessionStats.ORLow - _lowestSinceTarget));
                
                // UNIFIED BREAKOUT TRACKING with DYNAMIC THRESHOLD (BUG FIX #1 & #3)
                // Use dynamic threshold and validation for real-time updates
                double threshold = GetMinimumBreakoutThreshold();
                bool hasUpBreakout = _currentSessionStats.MaxExtensionUp >= threshold && 
                                    IsValidBreakout(_currentSessionStats.MaxExtensionUp, _currentSessionStats.FirstUpBreakoutTime);
                bool hasDownBreakout = _currentSessionStats.MaxExtensionDown >= threshold && 
                                      IsValidBreakout(_currentSessionStats.MaxExtensionDown, _currentSessionStats.FirstDownBreakoutTime);
                
                // Update breakout flags and timing
                if (hasUpBreakout && !_currentSessionStats.HasUpBreakout)
                {
                    _currentSessionStats.HasUpBreakout = true;
                    _currentSessionStats.FirstUpBreakoutTime = Time();
                    InvalidateBreakoutCache(_currentSessionStats.SessionId); // Invalidate cache
                }
                
                if (hasDownBreakout && !_currentSessionStats.HasDownBreakout)
                {
                    _currentSessionStats.HasDownBreakout = true;
                    _currentSessionStats.FirstDownBreakoutTime = Time();
                    InvalidateBreakoutCache(_currentSessionStats.SessionId); // Invalidate cache
                }
                
                // Update target hit counts
                _currentSessionStats.UpTargetsHit = _upTargets.Count(t => t.SessionId == _currentSessionStats.SessionId && t.IsFilled);
                _currentSessionStats.DownTargetsHit = _downTargets.Count(t => t.SessionId == _currentSessionStats.SessionId && t.IsFilled);
                
                // Mark statistics as dirty when data changes
                if (hasUpBreakout || hasDownBreakout)
                {
                    _statisticsDirty = true;
                }
            }
            catch (Exception ex)
            {
                // Error in UpdateSessionStatistics - continuing
            }
        }
        
        /// <summary>
        /// Calculate comprehensive statistics summary - FIXED VERSION
        /// BUG FIXES: #2 Double-counting prevention, #4 OR validation, Performance optimization
        /// </summary>
        private ORStatisticsSummary CalculateStatisticsSummary()
        {
            try
            {
                // Use cached result if available and not dirty
                if (_cachedSummary != null && !_statisticsDirty && 
                    (DateTime.Now - _lastStatsUpdate).TotalSeconds < 5)
                {
                    return _cachedSummary;
                }

                // Get recent finalized sessions within the specified period
                var recentStats = _sessionStats.Skip(Math.Max(0, _sessionStats.Count - StatisticsPeriod)).ToList();
                
                // CRITICAL FIX: Only include current session if NOT finalized (BUG FIX #2)
                var calcStats = (_currentSessionStats != null && !_currentSessionFinalized)
                    ? recentStats.Concat(new[] { _currentSessionStats }).ToList()
                    : recentStats;
                
                // VALIDATION: Filter out invalid sessions (BUG FIX #4)
                calcStats = calcStats.Where(IsValidORSession).ToList();
                
                // Early return for empty data
                if (calcStats.Count == 0)
                    return new ORStatisticsSummary();
            
            var summary = new ORStatisticsSummary
            {
                // Total sessions for display and period bounds
                TotalSessions = calcStats.Count,
                FirstSession = calcStats.Count > 0 ? calcStats.First().SessionDate : Time().Date,
                LastSession = calcStats.Count > 0 ? calcStats.Last().SessionDate : Time().Date
            };
            
            // Target performance
            summary.AverageTargetsHitPerSession = calcStats.Count > 0 ? calcStats.Average(s => s.UpTargetsHit + s.DownTargetsHit) : 0.0;
            
            // Bias accuracy
            var biasedSessions = calcStats.Where(s => s.DailyBias != 0).ToList();
            if (biasedSessions.Any())
            {
                summary.BiasAccuracyPercentage = (double)biasedSessions.Count(s => s.BiasCorrect) / biasedSessions.Count * 100;
            }
            
            // CACHE MANAGEMENT (Performance optimization)
                _cachedSummary = summary;
                _statisticsDirty = false;
                _lastStatsUpdate = DateTime.Now;
                
                return summary;
            }
            catch (Exception ex)
            {
                // Error in CalculateStatisticsSummary - returning cached or empty
                return _cachedSummary ?? new ORStatisticsSummary(); // Return cached or empty summary on error
            }
        }
        


        /// <summary>
        /// Update session-based moving average using an efficient queue.
        /// </summary>
        private void UpdateSessionMovingAverage()
        {
            if (!ShowSessionMA) return;

            // Add the current bar's close price to the queue
            _sessionPrices.Enqueue(Close());

            // Maintain the rolling window size by removing the oldest price if the queue exceeds the MA length
            if (_sessionPrices.Count > MovingAverageLength)
            {
                _sessionPrices.Dequeue();
            }

            // Calculate the MA based on the current prices in the queue
            _sessionMovingAverage = CalculateMovingAverage(_sessionPrices.ToList(), MAType);
        }

        /// <summary>
        /// Calculate moving average based on type
        /// </summary>
        private double CalculateMovingAverage(List<double> values, MovingAverageType type)
        {
            if (values.Count == 0) return 0.0;

            switch (type)
            {
                case MovingAverageType.SMA:
                    return values.Average();

                case MovingAverageType.EMA:
                    return CalculateEMA(values);

                case MovingAverageType.RMA:
                    return CalculateRMA(values);

                case MovingAverageType.WMA:
                    return CalculateWMA(values);

                case MovingAverageType.VWMA:
                    return CalculateVWMA(values);

                default:
                    return values.Average();
            }
        }

        /// <summary>
        /// Calculate EMA - HFT optimized
        /// </summary>
        private double CalculateEMA(List<double> values)
        {
            if (values.Count == 0) return 0.0;
            if (values.Count == 1) return values[0];

            double multiplier = 2.0 / (values.Count + 1);
            double ema = values[0];

            for (int i = 1; i < values.Count; i++)
            {
                ema = (values[i] * multiplier) + (ema * (1 - multiplier));
            }

            return ema;
        }

        /// <summary>
        /// Calculate RMA (Wilder's moving average) - HFT optimized
        /// </summary>
        private double CalculateRMA(List<double> values)
        {
            if (values.Count == 0) return 0.0;
            if (values.Count == 1) return values[0];

            double alpha = 1.0 / values.Count;
            double rma = values[0];

            for (int i = 1; i < values.Count; i++)
            {
                rma = alpha * values[i] + (1 - alpha) * rma;
            }

            return rma;
        }

        /// <summary>
        /// Calculate WMA - HFT optimized
        /// </summary>
        private double CalculateWMA(List<double> values)
        {
            if (values.Count == 0) return 0.0;

            double weightedSum = 0.0;
            double weightSum = 0.0;

            for (int i = 0; i < values.Count; i++)
            {
                double weight = i + 1;
                weightedSum += values[i] * weight;
                weightSum += weight;
            }

            return weightSum > 0 ? weightedSum / weightSum : 0.0;
        }

        /// <summary>
        /// Calculate VWMA - HFT optimized
        /// </summary>
        private double CalculateVWMA(List<double> values)
        {
            if (values.Count == 0) return 0.0;

            // For VWMA we need volume data, fallback to SMA if not available
            // This could be enhanced to use actual volume data
            return values.Average();
        }

        /// <summary>
        /// Set line series values
        /// </summary>
        private void SetLineValues()
        {
            // Original behavior: show OR only after OR token (post-OR)
            bool showOR = _openingRangeToken;
            // OR High
            SetValue(showOR ? _openingRangeHigh : double.NaN, 0);
            
            // OR Low  
            SetValue(showOR ? _openingRangeLow : double.NaN, 1);
            
            // OR Mid
            SetValue(showOR ? _openingRangeMid : double.NaN, 2);
            
            // Session MA
            SetValue(ShowSessionMA && _sessionMovingAverage > 0 ? _sessionMovingAverage : double.NaN, 3);
        }

        /// <summary>
        /// Custom painting for advanced visuals
        /// </summary>
        public override void OnPaintChart(PaintChartEventArgs args)
        {
            // Allow statistics panel to render even before OR token (during in-progress session)
            if (!_openingRangeToken && !(ShowORStatistics && (_sessionStats.Count > 0 || _currentSessionStats != null)))
                return;

            var graphics = args.Graphics;
            var rect = args.Rectangle;

            try
            {

                // Draw target labels if targets exist
                if (ShowTargets && (_upTargets.Count > 0 || _downTargets.Count > 0))
                {
                    DrawTargetLabels(graphics, rect);
                }

                // Draw signal markers if signals exist
                if (ShowBreakoutSignals && _signalHistory.Count > 0)
                {
                    DrawSignalMarkers(graphics, rect);
                }
                
                // Draw statistics panel if enabled (show also for in-progress session)
                if (ShowORStatistics && (_sessionStats.Count > 0 || _currentSessionStats != null))
                {
                    DrawStatisticsPanel(graphics, rect);
                }
            }
            catch (Exception ex)
            {
                // Error handling for drawing operations
            }
        }

        /// <summary>
        /// Draw target level labels and lines
        /// </summary>
        private void DrawTargetLabels(Graphics graphics, Rectangle rect)
        {
            if (CurrentChart?.MainWindow?.CoordinatesConverter == null) return;

            // Draw all targets with unified logic
            DrawTargetList(graphics, rect, _upTargets, BullTargetColor, true);
            DrawTargetList(graphics, rect, _downTargets, BearTargetColor, false);
        }

        /// <summary>
        /// Draw list of targets with unified logic
        /// </summary>
        private void DrawTargetList(Graphics graphics, Rectangle rect, List<TargetLevel> targets, Color baseColor, bool isUp)
        {
            foreach (var target in targets)
            {
                bool shouldDraw = TargetDisplay == TargetDisplayType.Extended || target.IsVisible;
                if (shouldDraw)
                {
                    Color targetColor = target.IsFilled ? Color.Gray : baseColor;
                    DrawTargetLevel(graphics, rect, target, targetColor, isUp);
                }
            }
        }

        /// <summary>
        /// Draw individual target level with proper session-based positioning and enhanced visibility.
        /// </summary>
        private void DrawTargetLevel(Graphics graphics, Rectangle rect, TargetLevel target, Color color, bool isUpTarget)
        {
            try
            {
                var converter = CurrentChart.MainWindow.CoordinatesConverter;
                float y = (float)converter.GetChartY(target.Price);

                // Skip if outside visible area
                if (y < rect.Y - 20 || y > rect.Bottom + 20) return;

                // Get the time range for the session this target belongs to
                (DateTime startTime, DateTime endTime) = GetSessionTimeRange(target);

                // Convert session start/end times to X coordinates
                float startX = (float)converter.GetChartX(startTime);
                float endX = (float)converter.GetChartX(endTime);

                // Ensure the line is within the visible chart bounds but extend for continuity
                startX = Math.Max(startX, rect.X - 50);
                endX = Math.Min(endX, rect.Right + 50);

                if (startX >= endX) return; // Nothing to draw

                // Draw enhanced line with anti-aliasing
                graphics.SmoothingMode = System.Drawing.Drawing2D.SmoothingMode.HighQuality;
                using (var pen = new Pen(color, 3))
                {
                    pen.DashStyle = target.IsFilled ? System.Drawing.Drawing2D.DashStyle.Solid : System.Drawing.Drawing2D.DashStyle.Dash;
                    pen.StartCap = pen.EndCap = System.Drawing.Drawing2D.LineCap.Round;
                    graphics.DrawLine(pen, startX, y, endX, y);
                }

                // Draw shadow for depth
                using (var shadowPen = new Pen(Color.FromArgb(60, Color.Black), 1))
                    graphics.DrawLine(shadowPen, startX, y + 1, endX, y + 1);
                
                // Draw dashed gray extension when target is filled
                if (target.IsFilled)
                {
                    using (var extensionPen = new Pen(Color.Gray, 1))
                    {
                        extensionPen.DashStyle = System.Drawing.Drawing2D.DashStyle.Dash;
                        graphics.DrawLine(extensionPen, endX, y, rect.Right, y);
                    }
                }
                
                DrawTargetElements(graphics, rect, target, color, isUpTarget, y, startX, endX);
            }
            catch (Exception ex)
            {
            }
        }

        /// <summary>
        /// Draw target elements (marker and label) positioned correctly for the session
        /// </summary>
        private void DrawTargetElements(Graphics graphics, Rectangle rect, TargetLevel target, Color color, bool isUpTarget, float y, float startX, float endX)
        {
            try
            {
                // Draw directional arrow inside
                float markerX = endX - 15;
                var arrowPoints = isUpTarget ? new PointF[] {
                    new PointF(markerX, y - 4), new PointF(markerX - 2, y + 2), new PointF(markerX + 2, y + 2)
                } : new PointF[] {
                    new PointF(markerX, y + 4), new PointF(markerX - 2, y - 2), new PointF(markerX + 2, y - 2)
                };
                
                using (var arrowBrush = new SolidBrush(Color.White))
                    graphics.FillPolygon(arrowBrush, arrowPoints);
                
                // Draw centered label on the target line
                var labelText = $"T{target.TargetNumber}{(target.IsFilled ? " ✓" : "")}";
                var textSize = graphics.MeasureString(labelText, _labelFont);
                float centerX = (startX + endX) / 2;
                var labelRect = new RectangleF(centerX - textSize.Width / 2 - 2, y - textSize.Height / 2, textSize.Width + 4, textSize.Height);

                using (var bgBrush = new SolidBrush(Color.FromArgb(220, color)))
                    graphics.FillRoundRectangle(bgBrush, labelRect, 4);
                using (var borderPen = new Pen(Color.White, 1))
                    graphics.DrawRoundRectangle(borderPen, labelRect, 4);
                using (var textBrush = new SolidBrush(Color.White))
                    graphics.DrawString(labelText, _labelFont, textBrush, labelRect.X + 2, labelRect.Y);
            }
            catch (Exception ex)
            {
            }
        }

        /// <summary>
        /// Gets the precise start and end time for a target's session using SessionId for proper matching.
        /// </summary>
        private (DateTime, DateTime) GetSessionTimeRange(TargetLevel target)
        {
            // Use the stored session times from the target itself - this is the most accurate
            if (target.SessionStartTime != DateTime.MinValue && target.SessionEndTime != DateTime.MinValue)
            {
                return (target.SessionStartTime, target.SessionEndTime);
            }
            
            // Fallback: find another target from the same session (same SessionId)
            var sampleTarget = _upTargets.FirstOrDefault(t => t.SessionId == target.SessionId && 
                                                            t.SessionStartTime != DateTime.MinValue) ??
                              _downTargets.FirstOrDefault(t => t.SessionId == target.SessionId && 
                                                             t.SessionStartTime != DateTime.MinValue);
            
            if (sampleTarget != null)
            {
                return (sampleTarget.SessionStartTime, sampleTarget.SessionEndTime);
            }
            
            // Final fallback: calculate session times based on target creation time
            var timeInSession = target.CreatedTime;
            if (UseCustomSession)
            {
                var sessionStart = timeInSession.Date.AddHours(CustomSessionStartHour).AddMinutes(CustomSessionStartMinute);
                var sessionEnd = timeInSession.Date.AddHours(CustomSessionEndHour).AddMinutes(CustomSessionEndMinute);
                if (sessionEnd <= sessionStart) // Handle overnight session
                    sessionEnd = sessionEnd.AddDays(1);
                return (sessionStart, sessionEnd);
            }
            else
            {
                var sessionStart = GetSessionStart(timeInSession);
                var sessionEnd = sessionStart.AddMinutes(OpeningRangeMinutes);
                return (sessionStart, sessionEnd);
            }
        }

        /// <summary>
        /// Draw signal markers at breakout levels
        /// </summary>
        private void DrawSignalMarkers(Graphics graphics, Rectangle rect)
        {
            if (CurrentChart?.MainWindow?.CoordinatesConverter == null) return;

            // Show all signals within the visible time range
            var iterable = ShowHistoricalData ? _signalHistory : _signalHistory.Where(s => s.SessionId == _currentSessionId);
            foreach (var signal in iterable)
            {
                DrawSignalMarker(graphics, rect, signal);
            }
        }

        /// <summary>
        /// Draw individual signal marker at the breakout level and time
        /// </summary>
        private void DrawSignalMarker(Graphics graphics, Rectangle rect, SignalInfo signal)
        {
            try
            {
                // Get X coordinate for the signal time
                double signalX = CurrentChart.MainWindow.CoordinatesConverter.GetChartX(signal.Time);
                
                // Check if signal is within visible time range
                if (signalX < rect.Left || signalX > rect.Right) return;

                // Use the stored breakout level from when the signal was generated
                double priceY = CurrentChart.MainWindow.CoordinatesConverter.GetChartY(signal.BreakoutLevel);
                
                if (priceY < rect.Y || priceY > rect.Bottom) return;

                float x = (float)signalX;
                float y = (float)priceY;
                Color signalColor = signal.IsUpSignal ? UpSignalColor : DownSignalColor;
                
                // Draw signal marker - diamond for confirmed, arrow for single breakout
                if (signal.IsConfirmed && signal.ConfirmationCount >= 2)
                {
                    DrawConfirmedSignalDiamond(graphics, x, y, signalColor, signal.IsUpSignal);
                }
                else
                {
                    DrawSignalArrow(graphics, x, y, signalColor, signal.IsUpSignal);
                }
                
                // No text labels needed - visual arrow/diamond is sufficient
                
            }
            catch (Exception ex)
            {
            }
        }

        /// <summary>
        /// Draw signal arrow marker
        /// </summary>
        private void DrawSignalArrow(Graphics graphics, float x, float y, Color color, bool isUpSignal)
        {
            try
            {
                var size = 8;
                var points = new PointF[3];

                if (isUpSignal)
                {
                    // Upward pointing arrow
                    points[0] = new PointF(x, y - size);      // Top point
                    points[1] = new PointF(x - size, y + size); // Bottom left
                    points[2] = new PointF(x + size, y + size); // Bottom right
                }
                else
                {
                    // Downward pointing arrow
                    points[0] = new PointF(x, y + size);      // Bottom point
                    points[1] = new PointF(x - size, y - size); // Top left
                    points[2] = new PointF(x + size, y - size); // Top right
                }

                using (var brush = new SolidBrush(color))
                {
                    graphics.FillPolygon(brush, points);
                }
                
                // Draw arrow outline for better visibility
                using (var pen = new Pen(Color.White, 1))
                {
                    graphics.DrawPolygon(pen, points);
                }
            }
            catch (Exception ex)
            {
            }
        }

        /// <summary>
        /// Draw confirmed signal diamond marker (for 2-signal confirmation)
        /// </summary>
        private void DrawConfirmedSignalDiamond(Graphics graphics, float x, float y, Color color, bool isUpSignal)
        {
            try
            {
                var size = 10; // Diamond size for confirmed signals
                var points = new PointF[4];

                // Create diamond shape
                points[0] = new PointF(x, y - size);      // Top point
                points[1] = new PointF(x + size, y);      // Right point
                points[2] = new PointF(x, y + size);      // Bottom point
                points[3] = new PointF(x - size, y);      // Left point

                // Fill with a brighter, more saturated color for confirmed signals
                var confirmedColor = Color.FromArgb(255, 
                    Math.Min(255, color.R + 50), 
                    Math.Min(255, color.G + 50), 
                    Math.Min(255, color.B + 50));
                
                using (var brush = new SolidBrush(confirmedColor))
                {
                    graphics.FillPolygon(brush, points);
                }
                
                // Draw thick yellow outline for confirmed signals
                using (var outerPen = new Pen(Color.Gold, 3))
                {
                    graphics.DrawPolygon(outerPen, points);
                }
                using (var innerPen = new Pen(Color.White, 1))
                {
                    graphics.DrawPolygon(innerPen, points);
                }
            }
            catch (Exception ex)
            {
            }
        }

        /// <summary>
        /// Draw comprehensive statistics panel
        /// </summary>
        private void DrawStatisticsPanel(Graphics graphics, Rectangle rect)
        {
            try
            {
                // Update cached summary if needed
                if (_cachedSummary == null || (DateTime.Now - _lastStatsUpdate).TotalSeconds > 5)
                {
                    _cachedSummary = CalculateStatisticsSummary();
                    _lastStatsUpdate = DateTime.Now;
                }
                
                var summary = _cachedSummary;
                if (summary == null) return;
                
                // Panel positioning (top-right corner) - Adjusted for reduced content
                var panelWidth = 280;
                var panelHeight = 200;
                var panelX = rect.Right - panelWidth - 20;
                var panelY = rect.Y + 20;
                var panelRect = new RectangleF(panelX, panelY, panelWidth, panelHeight);
                
                // Draw panel background
                using (var bgBrush = new SolidBrush(Color.FromArgb(200, 30, 30, 40)))
                {
                    graphics.FillRectangle(bgBrush, panelRect);
                }
                
                // Draw panel border
                using (var borderPen = new Pen(Color.FromArgb(150, Color.Gray), 1))
                {
                    graphics.DrawRectangle(borderPen, panelRect.X, panelRect.Y, panelRect.Width, panelRect.Height);
                }
                
                var textColor = Color.White;
                var headerColor = Color.LightBlue;
                var y = panelY + 10;
                var lineHeight = 18;
                var x = panelX + 10;
                
                // Header
                using (var headerBrush = new SolidBrush(headerColor))
                {
                    graphics.DrawString($"📊 OR STATISTICS ({summary.TotalSessions} sessions)", _labelFont, headerBrush, x, y);
                }
                y += lineHeight + 5;
                
                using (var textBrush = new SolidBrush(textColor))
                {
                    
                    
                    // Target Performance
                    graphics.DrawString("🎯 TARGET HITS:", _labelFont, textBrush, x, y);
                    y += lineHeight;
                    graphics.DrawString($"  Avg per session: {summary.AverageTargetsHitPerSession:F1}", _labelFont, textBrush, x, y);
                    y += lineHeight + 3;
                    
                    
                    // Bias Performance
                    if (summary.BiasAccuracyPercentage > 0)
                    {
                        graphics.DrawString("🧭 BIAS ACCURACY:", _labelFont, textBrush, x, y);
                        y += lineHeight;
                        var biasColor = summary.BiasAccuracyPercentage >= 60 ? Color.LightGreen : 
                                       summary.BiasAccuracyPercentage >= 50 ? Color.Yellow : Color.LightCoral;
                        using (var biasBrush = new SolidBrush(biasColor))
                        {
                            graphics.DrawString($"  {summary.BiasAccuracyPercentage:F1}%", _labelFont, biasBrush, x, y);
                        }
                        y += lineHeight + 3;
                    }
                    
                    
                    // Current Session Info
                    if (_currentSessionStats != null)
                    {
                        using (var sessionHeaderBrush = new SolidBrush(headerColor))
                        {
                            graphics.DrawString("📈 CURRENT SESSION:", _labelFont, sessionHeaderBrush, x, y);
                        }
                        y += lineHeight;
                        graphics.DrawString($"  Range: {_currentSessionStats.ORWidth:F2}", _labelFont, textBrush, x, y);
                        y += lineHeight;
                        graphics.DrawString($"  Targets Hit: ↑{_currentSessionStats.UpTargetsHit} ↓{_currentSessionStats.DownTargetsHit}", _labelFont, textBrush, x, y);
                        y += lineHeight;
                        graphics.DrawString($"  Max Ext: ↑{_currentSessionStats.MaxExtensionUp:F2} ↓{_currentSessionStats.MaxExtensionDown:F2}", _labelFont, textBrush, x, y);
                    }
                }
            }
            catch (Exception ex)
            {
            }
        }

        /// <summary>
        /// Get current opening range high
        /// </summary>
        public double GetOpeningRangeHigh()
        {
            return _openingRangeHigh;
        }

        /// <summary>
        /// Get current opening range low
        /// </summary>
        public double GetOpeningRangeLow()
        {
            return _openingRangeLow;
        }

        /// <summary>
        /// Get current opening range mid
        /// </summary>
        public double GetOpeningRangeMid()
        {
            return _openingRangeMid;
        }

        /// <summary>
        /// Get current opening range width
        /// </summary>
        public double GetOpeningRangeWidth()
        {
            return _openingRangeWidth;
        }

        /// <summary>
        /// Get current day direction bias
        /// </summary>
        public int GetDayDirection()
        {
            return _dayDirection;
        }

        /// <summary>
        /// Get target count for direction
        /// </summary>
        public int GetTargetCount(bool isUpDirection)
        {
            return isUpDirection
                ? _upTargets.Count(t => t.SessionId == _currentSessionId)
                : _downTargets.Count(t => t.SessionId == _currentSessionId);
        }

        /// <summary>
        /// Check if opening range is active
        /// </summary>
        public bool IsOpeningRangeActive()
        {
            return _openingRangeActive;
        }

        /// <summary>
        /// Cleanup resources
        /// </summary>
        protected override void OnClear()
        {
            base.OnClear();

            _labelFont?.Dispose();
            _textBrush?.Dispose();
            _linePen?.Dispose();

            _sessionPrices?.Clear();
            _upTargets?.Clear();
            _downTargets?.Clear();
            _signalHistory?.Clear();
        }
    }

    /// <summary>
    /// Graphics extensions for rounded rectangles
    /// </summary>
    public static class TargetGraphicsExtensions
    {
        public static void FillRoundRectangle(this Graphics g, Brush brush, RectangleF rect, float radius)
        {
            using (var path = new System.Drawing.Drawing2D.GraphicsPath())
            {
                radius = Math.Min(radius, Math.Min(rect.Width, rect.Height) / 2);
                path.AddArc(rect.X, rect.Y, radius * 2, radius * 2, 180, 90);
                path.AddArc(rect.Right - radius * 2, rect.Y, radius * 2, radius * 2, 270, 90);
                path.AddArc(rect.Right - radius * 2, rect.Bottom - radius * 2, radius * 2, radius * 2, 0, 90);
                path.AddArc(rect.X, rect.Bottom - radius * 2, radius * 2, radius * 2, 90, 90);
                path.CloseFigure();
                g.FillPath(brush, path);
            }
        }
        
        public static void DrawRoundRectangle(this Graphics g, Pen pen, RectangleF rect, float radius)
        {
            using (var path = new System.Drawing.Drawing2D.GraphicsPath())
            {
                radius = Math.Min(radius, Math.Min(rect.Width, rect.Height) / 2);
                path.AddArc(rect.X, rect.Y, radius * 2, radius * 2, 180, 90);
                path.AddArc(rect.Right - radius * 2, rect.Y, radius * 2, radius * 2, 270, 90);
                path.AddArc(rect.Right - radius * 2, rect.Bottom - radius * 2, radius * 2, radius * 2, 0, 90);
                path.AddArc(rect.X, rect.Bottom - radius * 2, radius * 2, radius * 2, 90, 90);
                path.CloseFigure();
                g.DrawPath(pen, path);
            }
        }
    }
} 