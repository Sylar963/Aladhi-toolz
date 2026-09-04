using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Drawing.Text;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using TradingPlatform.BusinessLayer;

namespace TradingSessions;

public enum RiskClockTimeZone
{
    NewYork,
    Utc,
    Local
}

/// <summary>
/// Account-level risk monitor. Monetary values come only from broker-provided
/// Trade.NetPnl and Position.NetPnL values for the selected account.
/// </summary>
public sealed class SessionAngel : Indicator
{
    private readonly object stateLock = new();
    private readonly Dictionary<string, TradeLedgerEntry> tradeLedger = new(StringComparer.Ordinal);
    private readonly Dictionary<Trade, Action> observedTrades = new();
    private readonly HashSet<Position> observedPositions = new();
    private readonly RiskAccumulator risk = new();

    private Timer? refreshTimer;
    private int refreshInProgress;
    private bool disposed;
    private bool needsHistoryReload;

    private DateTime riskDayStartUtc;
    private DateTime previousRiskDayStartUtc;
    private double realizedPnl;
    private double unrealizedPnl;
    private double currentSessionPnl;
    private double previousSessionRealizedPnl;
    private int closedExecutions;
    private int winningExecutions;
    private int losingExecutions;
    private double largestWin;
    private double largestLoss;

    private bool historyLoaded;
    private bool warningActive;
    private bool drawdownActive;
    private bool dangerActive;
    private bool accountLockRequested;
    private bool positionSizeWarningActive;
    private bool positionMaxSizeActive;
    private bool positionLossWarningActive;
    private bool positionLossDangerActive;
    private string dataStatus = "Select an account";
    private string currentActiveSession = "None";
    private readonly RecoveryMapState recoveryMap = new();

    private Font? headerFont;
    private Font? bodyFont;
    private Font? warningFont;
    private Font? statsFont;

    [InputParameter("Account", 10)]
    public Account? SelectedAccount { get; set; }

    [InputParameter("Daily loss limit", 20, 1, 1_000_000_000, 1, 2)]
    public double DailyLossLimit { get; set; } = 500.0;

    [InputParameter("Warning threshold", 30, 1, 1_000_000_000, 1, 2)]
    public double WarningThreshold { get; set; } = 300.0;

    [InputParameter("Maximum drawdown limit", 40, 1, 1_000_000_000, 1, 2)]
    public double MaxDrawdownLimit { get; set; } = 200.0;

    [InputParameter("Risk day time zone", 50)]
    public RiskClockTimeZone RiskTimeZone { get; set; } = RiskClockTimeZone.NewYork;

    [InputParameter("Risk day start hour", 60, 0, 23, 1, 0)]
    public int RiskDayStartHour { get; set; } = 9;

    [InputParameter("Risk day start minute", 70, 0, 59, 1, 0)]
    public int RiskDayStartMinute { get; set; } = 30;

    [InputParameter("Lock account at daily loss", 80)]
    public bool LockAccountOnDanger { get; set; }

    [InputParameter("Enable audio alerts", 90)]
    public bool EnableAudioAlerts { get; set; } = true;

    [InputParameter("Warning reset level (%)", 100, 10, 99, 1, 0)]
    public int WarningResetPercent { get; set; } = 80;

    [InputParameter("Show session statistics", 110)]
    public bool ShowSessionStats { get; set; } = true;

    [InputParameter("Show risk panel", 120)]
    public bool ShowRiskPanel { get; set; } = true;

    [InputParameter("Show performance metrics", 130)]
    public bool ShowPerformanceMetrics { get; set; } = true;

    [InputParameter("Show position recovery map", 140)]
    public bool ShowRecoveryMap { get; set; } = true;

    [InputParameter("Position loss cap", 150, 1, 1_000_000_000, 1, 2)]
    public double PositionLossCap { get; set; } = 500.0;

    [InputParameter("Position profit target", 160, 1, 1_000_000_000, 1, 2)]
    public double PositionProfitTarget { get; set; } = 500.0;

    [InputParameter("Working size warning", 170, 1, 1_000, 1, 0)]
    public int WorkingSizeWarning { get; set; } = 6;

    [InputParameter("Maximum position contracts", 180, 1, 1_000, 1, 0)]
    public int MaximumPositionContracts { get; set; } = 10;

    [InputParameter("Projected add contracts", 190, 0, 1_000, 1, 0)]
    public int ProjectedAddContracts { get; set; } = 5;

    [InputParameter("Position loss warning (%)", 195, 10, 99, 1, 0)]
    public int PositionLossWarningPercent { get; set; } = 80;

    [InputParameter("Warning color", 200)]
    public Color WarningColor { get; set; } = Color.Orange;

    [InputParameter("Danger color", 210)]
    public Color DangerColor { get; set; } = Color.Red;

    [InputParameter("Safe color", 220)]
    public Color SafeColor { get; set; } = Color.LimeGreen;

    [InputParameter("Panel background", 230)]
    public Color PanelBackground { get; set; } = Color.FromArgb(210, 30, 30, 40);

    [InputParameter("Animation speed", 240, 1, 10, 1, 0)]
    public int AnimationSpeed { get; set; } = 5;

    public SessionAngel()
    {
        Name = "Session Angel";
        Description = "Account risk monitor with a live position recovery and Fibonacci map";
        AddLineSeries("State", Color.Transparent, 1, LineStyle.Solid);
        SeparateWindow = false;
    }

    protected override void OnInit()
    {
        disposed = false;
        InitializeGraphicsResources();
        ResetForRiskDay(DateTime.UtcNow);

        Core.Instance.TradeAdded += OnTradeAdded;
        Core.Instance.PositionAdded += OnPositionAdded;
        Core.Instance.PositionRemoved += OnPositionRemoved;
        foreach (Position position in Core.Instance.Positions)
            SubscribeToPosition(position);

        LoadTradeHistory();
        RefreshAccountSnapshot(false);
        refreshTimer = new Timer(OnRefreshTimer, null, TimeSpan.FromSeconds(1), TimeSpan.FromSeconds(1));
    }

    protected override void OnUpdate(UpdateArgs args)
    {
        SetValue(double.NaN);
        if (args.Reason != UpdateReason.HistoricalBar)
            RefreshAccountSnapshot(true);
    }

    private void OnRefreshTimer(object? state)
    {
        if (!disposed)
            RefreshAccountSnapshot(true);
    }

    private void OnTradeAdded(Trade trade)
    {
        if (disposed || !IsSelectedAccount(trade.Account))
            return;

        SubscribeToTrade(trade);
        EnsureCurrentRiskDay(DateTime.UtcNow);
        lock (stateLock)
        {
            StoreTradeLocked(trade);
            RecalculateTradeMetricsLocked();
        }
        RefreshAccountSnapshot(true);
    }

    private void OnTradeUpdated(Trade trade)
    {
        if (disposed || !IsSelectedAccount(trade.Account))
            return;

        lock (stateLock)
        {
            StoreTradeLocked(trade);
            RecalculateTradeMetricsLocked();
        }
        RefreshAccountSnapshot(true);
    }

    private void SubscribeToTrade(Trade trade)
    {
        Action handler;
        lock (stateLock)
        {
            if (observedTrades.ContainsKey(trade))
                return;
            handler = () => OnTradeUpdated(trade);
            observedTrades.Add(trade, handler);
        }
        trade.Updated += handler;
    }

    private void OnPositionAdded(Position position)
    {
        SubscribeToPosition(position);
        if (!disposed && IsSelectedAccount(position.Account))
            RefreshAccountSnapshot(true);
    }

    private void OnPositionRemoved(Position position)
    {
        UnsubscribeFromPosition(position);
        if (!disposed && IsSelectedAccount(position.Account))
            RefreshAccountSnapshot(true);
    }

    private void OnPositionUpdated(Position position)
    {
        if (!disposed && IsSelectedAccount(position.Account))
            RefreshAccountSnapshot(true);
    }

    private void SubscribeToPosition(Position position)
    {
        lock (stateLock)
        {
            if (!observedPositions.Add(position))
                return;
        }
        position.Updated += OnPositionUpdated;
    }

    private void UnsubscribeFromPosition(Position position)
    {
        bool removed;
        lock (stateLock)
            removed = observedPositions.Remove(position);
        if (removed)
            position.Updated -= OnPositionUpdated;
    }

    private void RefreshAccountSnapshot(bool allowHistoryReload)
    {
        if (disposed || Interlocked.Exchange(ref refreshInProgress, 1) != 0)
            return;

        AlertKind alert = AlertKind.None;
        bool shouldLock = false;

        try
        {
            Account? account = SelectedAccount;
            if (account is null)
            {
                lock (stateLock)
                {
                    dataStatus = "Select an account";
                    historyLoaded = false;
                    warningActive = drawdownActive = dangerActive = false;
                    positionSizeWarningActive = positionMaxSizeActive = false;
                    positionLossWarningActive = positionLossDangerActive = false;
                    recoveryMap.Reset();
                    realizedPnl = unrealizedPnl = currentSessionPnl = 0;
                }
                return;
            }

            EnsureCurrentRiskDay(DateTime.UtcNow);
            if (allowHistoryReload && needsHistoryReload)
                LoadTradeHistory();

            double openPnl = CalculateOpenPositionPnl(account);
            PositionAggregate? chartPosition = CaptureChartPosition(account);
            lock (stateLock)
            {
                unrealizedPnl = openPnl;
                currentSessionPnl = realizedPnl + unrealizedPnl;
                risk.Update(currentSessionPnl);
                currentActiveSession = GetActiveSessions(DateTime.UtcNow);
                EvaluateRiskLocked(out alert, out shouldLock);
                AlertKind positionAlert = UpdateRecoveryMapLocked(chartPosition);
                alert = HigherPriorityAlert(alert, positionAlert);

                string readiness = historyLoaded ? "Broker data ready" : "Partial data";
                string configuration = WarningThreshold >= DailyLossLimit
                    ? "; warning clamped below limit"
                    : string.Empty;
                dataStatus = readiness + configuration;
            }
        }
        catch (Exception ex)
        {
            lock (stateLock)
            {
                historyLoaded = false;
                dataStatus = "Partial data: " + ex.Message;
            }
            Debug.WriteLine($"Session Angel refresh error: {ex}");
        }
        finally
        {
            Interlocked.Exchange(ref refreshInProgress, 0);
        }

        TriggerAlert(alert);
        if (shouldLock)
            LockSelectedAccount();
        RequestChartRedraw();
    }

    private void RequestChartRedraw()
    {
        try
        {
            CurrentChart?.RedrawBuffer();
        }
        catch (Exception ex)
        {
            Debug.WriteLine($"Session Angel chart redraw error: {ex.Message}");
        }
    }

    private void EnsureCurrentRiskDay(DateTime utcNow)
    {
        DateTime expected = GetRiskDayStartUtc(utcNow);
        lock (stateLock)
        {
            if (expected == riskDayStartUtc)
                return;
        }
        ResetForRiskDay(utcNow);
    }

    private void ResetForRiskDay(DateTime utcNow)
    {
        DateTime currentStart = GetRiskDayStartUtc(utcNow);
        DateTime previousStart = GetRiskDayStartUtc(currentStart.AddTicks(-1));

        lock (stateLock)
        {
            riskDayStartUtc = currentStart;
            previousRiskDayStartUtc = previousStart;
            tradeLedger.Clear();
            realizedPnl = unrealizedPnl = currentSessionPnl = 0;
            previousSessionRealizedPnl = 0;
            closedExecutions = winningExecutions = losingExecutions = 0;
            largestWin = largestLoss = 0;
            risk.Reset();
            historyLoaded = false;
            needsHistoryReload = SelectedAccount is not null;
            warningActive = drawdownActive = dangerActive = false;
            accountLockRequested = false;
            dataStatus = SelectedAccount is null ? "Select an account" : "Loading broker history";
        }
    }

    private void LoadTradeHistory()
    {
        Account? account = SelectedAccount;
        if (account is null)
            return;

        DateTime from;
        lock (stateLock)
            from = previousRiskDayStartUtc;

        try
        {
            var request = new TradesHistoryRequestParameters
            {
                From = from,
                To = DateTime.UtcNow,
                ForceReload = false
            };
            var trades = Core.Instance.GetTrades(request, account.ConnectionId);

            lock (stateLock)
            {
                foreach (Trade trade in trades.OrderBy(t => t.DateTime))
                    StoreTradeLocked(trade);
                RecalculateTradeMetricsLocked();
                historyLoaded = true;
                needsHistoryReload = false;
            }
        }
        catch (Exception ex)
        {
            lock (stateLock)
            {
                historyLoaded = false;
                needsHistoryReload = false;
                dataStatus = "Partial data: trade history unavailable";
            }
            Debug.WriteLine($"Session Angel history error: {ex}");
        }
    }

    private void StoreTradeLocked(Trade trade)
    {
        if (!IsSelectedAccount(trade.Account))
            return;

        string key = GetTradeKey(trade);
        tradeLedger[key] = new TradeLedgerEntry(
            NormalizeUtc(trade.DateTime),
            GetPnlValue(trade.NetPnl),
            trade.NetPnl is not null,
            trade.PositionImpactType == PositionImpactType.Close,
            trade.PositionId ?? string.Empty,
            trade.Symbol);
    }

    private void RecalculateTradeMetricsLocked()
    {
        realizedPnl = 0;
        previousSessionRealizedPnl = 0;
        closedExecutions = winningExecutions = losingExecutions = 0;
        largestWin = largestLoss = 0;

        foreach (TradeLedgerEntry trade in tradeLedger.Values)
        {
            if (trade.TimeUtc >= riskDayStartUtc)
            {
                realizedPnl += trade.Pnl;
                if (!trade.IsClose)
                    continue;

                closedExecutions++;
                if (!trade.HasPnl)
                    continue;

                if (trade.Pnl > 0)
                {
                    winningExecutions++;
                    largestWin = Math.Max(largestWin, trade.Pnl);
                }
                else if (trade.Pnl < 0)
                {
                    losingExecutions++;
                    largestLoss = Math.Min(largestLoss, trade.Pnl);
                }
            }
            else if (trade.TimeUtc >= previousRiskDayStartUtc)
            {
                previousSessionRealizedPnl += trade.Pnl;
            }
        }
    }

    private double CalculateOpenPositionPnl(Account account)
    {
        double total = 0;
        foreach (Position position in Core.Instance.Positions)
        {
            if (SameAccount(position.Account, account))
                total += GetPnlValue(position.NetPnL);
        }
        return total;
    }

    private PositionAggregate? CaptureChartPosition(Account account)
    {
        Symbol? chartSymbol = Symbol;
        if (chartSymbol is null)
            return null;

        Position[] matches = Core.Instance.Positions
            .Where(position => position.State == BusinessObjectState.Normal
                && double.IsFinite(position.Quantity)
                && Math.Abs(position.Quantity) > 0
                && SameAccount(position.Account, account)
                && SameSymbol(position.Symbol, chartSymbol))
            .ToArray();
        if (matches.Length == 0)
            return null;

        Side side = matches[0].Side;
        if (matches.Any(position => position.Side != side))
            return PositionAggregate.Hedged(chartSymbol, account);

        double quantity = matches.Sum(position => Math.Abs(position.Quantity));
        if (!double.IsFinite(quantity) || quantity <= 0)
            return null;

        double weightedEntry = matches.Sum(position => position.OpenPrice * Math.Abs(position.Quantity)) / quantity;
        double currentPrice = matches
            .Select(position => position.CurrentPrice)
            .FirstOrDefault(price => double.IsFinite(price) && price > 0);
        if (!double.IsFinite(currentPrice) || currentPrice <= 0)
            currentPrice = side == Side.Buy ? chartSymbol.Bid : chartSymbol.Ask;
        if (!double.IsFinite(currentPrice) || currentPrice <= 0)
            currentPrice = chartSymbol.Last;

        double netPnl = matches.Sum(position => GetPnlValue(position.NetPnL));
        Symbol riskSymbol = matches[0].Symbol;
        double tickSize = riskSymbol.GetTickSize(currentPrice);
        if (!double.IsFinite(tickSize) || tickSize <= 0)
            tickSize = chartSymbol.GetTickSize(currentPrice);
        if (!double.IsFinite(tickSize) || tickSize <= 0)
            tickSize = riskSymbol.TickSize;

        double tickCost = riskSymbol.GetTickCost(currentPrice);
        if (!double.IsFinite(tickCost) || tickCost == 0)
            tickCost = chartSymbol.GetTickCost(currentPrice);
        if ((!double.IsFinite(tickCost) || tickCost == 0) && tickSize > 0)
        {
            Position? inferencePosition = matches.FirstOrDefault(position =>
                Math.Abs(position.CurrentPrice - position.OpenPrice) >= tickSize
                && Math.Abs(position.Quantity) > 0
                && Math.Abs(GetPnlValue(position.GrossPnL)) > 0);
            if (inferencePosition is not null)
            {
                double priceMove = Math.Abs(inferencePosition.CurrentPrice - inferencePosition.OpenPrice);
                double grossPnl = Math.Abs(GetPnlValue(inferencePosition.GrossPnL));
                tickCost = grossPnl / priceMove / Math.Abs(inferencePosition.Quantity) * tickSize;
            }
        }

        DateTime openTime = matches.Min(position => NormalizeUtc(position.OpenTime));
        string[] positionIds = matches
            .SelectMany(position => new[] { position.Id, position.UniqueId })
            .Where(id => !string.IsNullOrWhiteSpace(id))
            .Distinct(StringComparer.Ordinal)
            .OrderBy(id => id, StringComparer.Ordinal)
            .ToArray();
        string positionIdentity = string.Join(",", positionIds);
        if (string.IsNullOrWhiteSpace(positionIdentity))
            positionIdentity = openTime.Ticks.ToString();
        string lifecycleKey = string.Join("|", account.ConnectionId, account.Id,
            chartSymbol.ConnectionId, chartSymbol.Id, side, positionIdentity);
        return new PositionAggregate(
            lifecycleKey,
            chartSymbol.Name,
            side,
            quantity,
            weightedEntry,
            currentPrice,
            netPnl,
            tickSize,
            tickCost,
            false,
            riskSymbol,
            positionIds,
            openTime);
    }

    private AlertKind UpdateRecoveryMapLocked(PositionAggregate? position)
    {
        if (position is null)
        {
            if (SelectedAccount is not null && Symbol is not null)
                recoveryMap.SetUnavailable(Symbol.Name, "Waiting for an open position on this chart");
            else
                recoveryMap.Reset();
            positionSizeWarningActive = positionMaxSizeActive = false;
            positionLossWarningActive = positionLossDangerActive = false;
            return AlertKind.None;
        }

        PositionAggregate p = position.Value;
        if (p.IsHedged)
        {
            recoveryMap.SetUnavailable(p.SymbolName, "Hedged long/short positions cannot share one recovery map");
            positionSizeWarningActive = positionMaxSizeActive = false;
            positionLossWarningActive = positionLossDangerActive = false;
            return AlertKind.None;
        }

        if (!p.IsUsable)
        {
            recoveryMap.SetUnavailable(p.SymbolName, "Waiting for broker position prices");
            return AlertKind.None;
        }

        double campaignRealizedPnl = CalculateCampaignRealizedPnlLocked(p);
        bool isNewLifecycle = !recoveryMap.Active
            || !string.Equals(recoveryMap.LifecycleKey, p.LifecycleKey, StringComparison.Ordinal)
            || recoveryMap.Side != p.Side;
        if (isNewLifecycle)
            recoveryMap.Start(p, campaignRealizedPnl);
        else
            recoveryMap.Update(p, campaignRealizedPnl);

        recoveryMap.CalculateLevels(
            Math.Max(1, PositionLossCap),
            Math.Max(1, PositionProfitTarget),
            Math.Max(0, ProjectedAddContracts),
            Math.Max(1, MaximumPositionContracts));

        double loss = Math.Max(0, -recoveryMap.CampaignNetPnl);
        double lossCap = Math.Max(1, PositionLossCap);
        bool nextLossDanger = loss >= lossCap;
        bool nextLossWarning = nextLossDanger
            || loss >= lossCap * Math.Clamp(PositionLossWarningPercent, 10, 99) / 100.0;
        bool nextMaxSize = p.Quantity >= Math.Max(1, MaximumPositionContracts);
        bool nextSizeWarning = nextMaxSize || p.Quantity >= Math.Max(1, WorkingSizeWarning);

        AlertKind alert = AlertKind.None;
        if (nextLossDanger && !positionLossDangerActive)
            alert = AlertKind.PositionLoss;
        else if (nextMaxSize && !positionMaxSizeActive)
            alert = AlertKind.MaxSize;
        else if ((nextLossWarning && !positionLossWarningActive)
                 || (nextSizeWarning && !positionSizeWarningActive))
            alert = AlertKind.PositionWarning;

        positionLossDangerActive = nextLossDanger;
        positionLossWarningActive = nextLossWarning;
        positionMaxSizeActive = nextMaxSize;
        positionSizeWarningActive = nextSizeWarning;
        return alert;
    }

    private double CalculateCampaignRealizedPnlLocked(PositionAggregate position)
    {
        double total = 0;
        foreach (TradeLedgerEntry trade in tradeLedger.Values)
        {
            if (!trade.IsClose || !trade.HasPnl || trade.TimeUtc < position.OpenTimeUtc)
                continue;

            bool hasPositionIdentity = !string.IsNullOrWhiteSpace(trade.PositionId)
                && position.PositionIds.Length > 0;
            bool matchesPosition = hasPositionIdentity
                ? position.PositionIds.Contains(trade.PositionId, StringComparer.Ordinal)
                : SameSymbol(trade.Instrument, position.Instrument);
            if (matchesPosition)
                total += trade.Pnl;
        }
        return total;
    }

    private void EvaluateRiskLocked(out AlertKind alert, out bool shouldLock)
    {
        alert = AlertKind.None;
        shouldLock = false;
        double dailyLimit = Math.Max(1, DailyLossLimit);
        double warningLimit = Math.Min(Math.Max(1, WarningThreshold), Math.Max(1, dailyLimit - 0.01));
        double currentLoss = Math.Max(0, -currentSessionPnl);

        if (!dangerActive && currentLoss >= dailyLimit)
        {
            dangerActive = true;
            warningActive = false;
            alert = AlertKind.Danger;
        }

        if (!drawdownActive && risk.MaxDrawdown >= Math.Max(1, MaxDrawdownLimit))
        {
            drawdownActive = true;
            if (!dangerActive)
                alert = AlertKind.Drawdown;
        }

        if (!dangerActive)
        {
            double reset = warningLimit * Math.Clamp(WarningResetPercent, 10, 99) / 100.0;
            bool lossWarning = warningActive ? currentLoss >= reset : currentLoss >= warningLimit;
            bool nextWarning = lossWarning || drawdownActive;
            if (nextWarning && !warningActive && alert == AlertKind.None)
                alert = AlertKind.Warning;
            warningActive = nextWarning;
        }

        if (dangerActive && LockAccountOnDanger && !accountLockRequested)
        {
            accountLockRequested = true;
            shouldLock = true;
        }
    }

    private void LockSelectedAccount()
    {
        Account? account = SelectedAccount;
        if (account is null)
            return;
        try
        {
            Core.Instance.LockAccount(account);
            lock (stateLock)
                dataStatus += "; account lock requested";
        }
        catch (Exception ex)
        {
            lock (stateLock)
                dataStatus += "; account lock failed";
            Debug.WriteLine($"Session Angel account lock error: {ex}");
        }
    }

    private void TriggerAlert(AlertKind alert)
    {
        if (!EnableAudioAlerts || alert == AlertKind.None)
            return;

        _ = Task.Run(() =>
        {
            try
            {
                (int frequency, int duration) = alert switch
                {
                    AlertKind.PositionLoss => (1150, 600),
                    AlertKind.MaxSize => (1050, 450),
                    AlertKind.PositionWarning => (900, 300),
                    AlertKind.Danger => (1000, 500),
                    AlertKind.Drawdown => (650, 400),
                    _ => (800, 250)
                };
                Console.Beep(frequency, duration);
            }
            catch (Exception ex)
            {
                Debug.WriteLine($"Session Angel audio error: {ex.Message}");
            }
        });
    }

    private DateTime GetRiskDayStartUtc(DateTime utcNow)
    {
        utcNow = NormalizeUtc(utcNow);
        TimeZoneInfo zone = GetRiskTimeZone();
        DateTime localNow = TimeZoneInfo.ConvertTimeFromUtc(utcNow, zone);
        DateTime localStart = new(
            localNow.Year,
            localNow.Month,
            localNow.Day,
            Math.Clamp(RiskDayStartHour, 0, 23),
            Math.Clamp(RiskDayStartMinute, 0, 59),
            0,
            DateTimeKind.Unspecified);
        if (localNow < localStart)
            localStart = localStart.AddDays(-1);
        while (zone.IsInvalidTime(localStart))
            localStart = localStart.AddMinutes(1);
        return TimeZoneInfo.ConvertTimeToUtc(localStart, zone);
    }

    private TimeZoneInfo GetRiskTimeZone()
    {
        return RiskTimeZone switch
        {
            RiskClockTimeZone.Utc => TimeZoneInfo.Utc,
            RiskClockTimeZone.Local => TimeZoneInfo.Local,
            _ => FindNewYorkTimeZone()
        };
    }

    private static TimeZoneInfo FindNewYorkTimeZone()
    {
        try
        {
            return TimeZoneInfo.FindSystemTimeZoneById("Eastern Standard Time");
        }
        catch (TimeZoneNotFoundException)
        {
            return TimeZoneInfo.FindSystemTimeZoneById("America/New_York");
        }
    }

    private static string GetActiveSessions(DateTime utcNow)
    {
        TimeSpan eastern = TimeZoneInfo.ConvertTimeFromUtc(NormalizeUtc(utcNow), FindNewYorkTimeZone()).TimeOfDay;
        var sessions = new List<string>(3);
        if (IsInSession(eastern, new TimeSpan(19, 0, 0), new TimeSpan(4, 0, 0))) sessions.Add("Tokyo");
        if (IsInSession(eastern, new TimeSpan(3, 0, 0), new TimeSpan(11, 0, 0))) sessions.Add("London");
        if (IsInSession(eastern, new TimeSpan(9, 30, 0), new TimeSpan(16, 0, 0))) sessions.Add("New York");
        return sessions.Count == 0 ? "None" : string.Join(" + ", sessions);
    }

    public static bool IsInSession(TimeSpan time, TimeSpan start, TimeSpan end)
        => end > start ? time >= start && time < end : time >= start || time < end;

    private bool IsSelectedAccount(Account? account)
        => SelectedAccount is not null && account is not null && SameAccount(account, SelectedAccount);

    private static bool SameAccount(Account left, Account right)
        => string.Equals(left.Id, right.Id, StringComparison.Ordinal)
           && string.Equals(left.ConnectionId, right.ConnectionId, StringComparison.Ordinal);

    private static bool SameSymbol(Symbol left, Symbol right)
    {
        if (string.Equals(left.UniqueId, right.UniqueId, StringComparison.Ordinal)
            || string.Equals(left.Id, right.Id, StringComparison.Ordinal)
            || string.Equals(left.Name, right.Name, StringComparison.OrdinalIgnoreCase))
            return true;

        return !string.IsNullOrWhiteSpace(left.Root)
            && string.Equals(left.Root, right.Root, StringComparison.OrdinalIgnoreCase)
            && left.ExpirationDate != default
            && right.ExpirationDate != default
            && left.ExpirationDate.Date == right.ExpirationDate.Date;
    }

    private static AlertKind HigherPriorityAlert(AlertKind left, AlertKind right)
        => AlertPriority(right) > AlertPriority(left) ? right : left;

    private static int AlertPriority(AlertKind alert)
        => alert switch
        {
            AlertKind.PositionLoss => 6,
            AlertKind.Danger => 5,
            AlertKind.MaxSize => 4,
            AlertKind.Drawdown => 3,
            AlertKind.PositionWarning => 2,
            AlertKind.Warning => 1,
            _ => 0
        };

    private static double GetPnlValue(PnLItem? pnl)
    {
        double value = pnl?.Value ?? 0;
        return double.IsFinite(value) ? value : 0;
    }

    private static DateTime NormalizeUtc(DateTime value)
    {
        return value.Kind switch
        {
            DateTimeKind.Utc => value,
            DateTimeKind.Local => value.ToUniversalTime(),
            _ => DateTime.SpecifyKind(value, DateTimeKind.Utc)
        };
    }

    private static string GetTradeKey(Trade trade)
    {
        if (!string.IsNullOrWhiteSpace(trade.Id))
            return trade.ConnectionId + ":" + trade.Id;
        return string.Join("|", trade.ConnectionId, trade.Account?.Id, trade.OrderId,
            trade.PositionId, NormalizeUtc(trade.DateTime).Ticks, trade.Price, trade.Quantity);
    }

    private void InitializeGraphicsResources()
    {
        DisposeFonts();
        try
        {
            headerFont = new Font(FontFamily.GenericSansSerif, 16, FontStyle.Bold);
            bodyFont = new Font(FontFamily.GenericSansSerif, 10, FontStyle.Regular);
            warningFont = new Font(FontFamily.GenericSansSerif, 15, FontStyle.Bold);
            statsFont = new Font(FontFamily.GenericMonospace, 9, FontStyle.Regular);
        }
        catch (Exception ex)
        {
            DisposeFonts();
            Debug.WriteLine($"Session Angel font error: {ex.Message}");
        }
    }

    public override void OnPaintChart(PaintChartEventArgs args)
    {
        DashboardSnapshot s = CaptureSnapshot();
        GraphicsState saved = args.Graphics.Save();
        try
        {
            args.Graphics.SmoothingMode = SmoothingMode.AntiAlias;
            args.Graphics.TextRenderingHint = TextRenderingHint.ClearTypeGridFit;
            args.Graphics.CompositingQuality = CompositingQuality.HighQuality;
            if (ShowRecoveryMap) DrawRecoveryMap(args, s);
            if (ShowRiskPanel) DrawMainPanel(args, s);
            if (ShowRecoveryMap) DrawRecoveryPanel(args, s);
            DrawStatusIndicator(args, s);
            float lowerY = args.Rectangle.Y + 282;
            if (ShowSessionStats && args.Rectangle.Height >= 430) DrawStatistics(args, s, lowerY);
            if (ShowPerformanceMetrics && args.Rectangle.Height >= 430 && args.Rectangle.Width >= 610) DrawPerformance(args, s, lowerY);
            DrawWarningOverlay(args, s);
        }
        catch (Exception ex)
        {
            Debug.WriteLine($"Session Angel drawing error: {ex}");
        }
        finally
        {
            args.Graphics.Restore(saved);
        }
    }

    private DashboardSnapshot CaptureSnapshot()
    {
        lock (stateLock)
        {
            Account? account = SelectedAccount;
            return new DashboardSnapshot(
                account?.Name ?? "No account selected",
                account?.AccountCurrency?.ToString() ?? string.Empty,
                dataStatus,
                currentActiveSession,
                realizedPnl,
                unrealizedPnl,
                currentSessionPnl,
                risk.SessionPeak,
                risk.MaximumAdverseExcursion,
                risk.MaxDrawdown,
                previousSessionRealizedPnl,
                closedExecutions,
                winningExecutions,
                losingExecutions,
                largestWin,
                largestLoss,
                historyLoaded,
                warningActive,
                dangerActive,
                accountLockRequested,
                recoveryMap.Capture(
                    positionSizeWarningActive,
                    positionMaxSizeActive,
                    positionLossWarningActive,
                    positionLossDangerActive));
        }
    }

    private void DrawRecoveryMap(PaintChartEventArgs args, DashboardSnapshot s)
    {
        RecoverySnapshot r = s.Recovery;
        var windows = CurrentChart?.Windows;
        if (!r.Active || windows is null || windows.Length == 0
            || args.WindowIndex < 0 || args.WindowIndex >= windows.Length)
            return;

        var levels = new List<RecoveryLine>(10);
        double minimumSeparation = Math.Max(r.TickSize / 2, 1e-9);
        AddRecoveryLine(levels, "MAX LOSS", r.HardLossPrice, r, DangerColor, DashStyle.Solid, 3, minimumSeparation);
        AddRecoveryLine(levels, $"+{Money(PositionProfitTarget, s.Currency)} TARGET", r.ProfitTargetPrice, r, SafeColor, DashStyle.Solid, 2, minimumSeparation);
        AddRecoveryLine(levels, "ENTRY / FIB 0", r.AverageEntryPrice, r, Color.DeepSkyBlue, DashStyle.Solid, 2, minimumSeparation);
        AddRecoveryLine(levels, "NET B/E AFTER COSTS", r.BreakEvenPrice, r, Color.White, DashStyle.Dash, 2, minimumSeparation);

        if (Math.Abs(r.AverageEntryPrice - r.WorstPrice) >= Math.Max(r.TickSize, 1e-9))
        {
            AddRecoveryLine(levels, "FIB 38.2%", r.Fib382Price, r, Color.Orange, DashStyle.Dash, 1, minimumSeparation);
            AddRecoveryLine(levels, "FIB 50%", r.Fib500Price, r, Color.Gold, DashStyle.Solid, 2, minimumSeparation);
            AddRecoveryLine(levels, "FIB 61.8%", r.Fib618Price, r, Color.YellowGreen, DashStyle.Dash, 1, minimumSeparation);
            AddRecoveryLine(levels, "WORST / FIB 100", r.WorstPrice, r, Color.IndianRed, DashStyle.Dot, 1, minimumSeparation);
        }

        var converter = windows[args.WindowIndex].CoordinatesConverter;
        Rectangle rect = args.Rectangle;
        foreach (RecoveryLine level in levels)
        {
            double rawY = converter.GetChartY(level.Price);
            if (!double.IsFinite(rawY) || rawY < rect.Top - 2 || rawY > rect.Bottom + 2)
                continue;

            float y = (float)rawY;
            using var pen = new Pen(Color.FromArgb(220, level.Color), level.Width) { DashStyle = level.DashStyle };
            args.Graphics.DrawLine(pen, rect.Left, y, rect.Right, y);

            string label = $"{level.Label}  {FormatChartPrice(level.Price)}  Net {SignedMoney(level.CampaignPnl, s.Currency)}";
            SizeF size = args.Graphics.MeasureString(label, StatsFont);
            float boxX = Math.Max(rect.Left + 4, rect.Right - size.Width - 12);
            float boxY = Math.Clamp(y - size.Height - 2, rect.Top + 2, rect.Bottom - size.Height - 4);
            var box = new RectangleF(boxX, boxY, size.Width + 8, size.Height + 3);
            using (var background = new SolidBrush(Color.FromArgb(205, 18, 18, 24)))
                args.Graphics.FillRectangle(background, box);
            using (var brush = new SolidBrush(level.Color))
                args.Graphics.DrawString(label, StatsFont, brush, box.X + 4, box.Y + 1);
        }
    }

    private static void AddRecoveryLine(
        List<RecoveryLine> levels,
        string label,
        double price,
        RecoverySnapshot recovery,
        Color color,
        DashStyle dashStyle,
        float width,
        double minimumSeparation)
    {
        if (!double.IsFinite(price) || price <= 0)
            return;

        int existingIndex = levels.FindIndex(level => Math.Abs(level.Price - price) < minimumSeparation);
        if (existingIndex >= 0)
        {
            RecoveryLine existing = levels[existingIndex];
            levels[existingIndex] = existing with { Label = existing.Label + " / " + label };
            return;
        }

        levels.Add(new RecoveryLine(label, price, recovery.CampaignPnlAt(price), color, dashStyle, width));
    }

    private void DrawRecoveryPanel(PaintChartEventArgs args, DashboardSnapshot s)
    {
        RecoverySnapshot r = s.Recovery;
        if (!r.Visible || args.Rectangle.Width < 460)
            return;

        Rectangle chart = args.Rectangle;
        float width = Math.Min(405, chart.Width - 32);
        float x = Math.Max(chart.X + 16, chart.Right - width - 16);
        var panel = new RectangleF(x, chart.Y + 112, width, r.Active ? 171 : 62);
        Color borderColor = r.LossDanger || r.MaxSize ? DangerColor
            : r.LossWarning || r.SizeWarning ? WarningColor
            : Color.DeepSkyBlue;
        using (var bg = new SolidBrush(Color.FromArgb(215, 20, 24, 32)))
            args.Graphics.FillRoundRectangle(bg, panel, 10);
        using (var pen = new Pen(Color.FromArgb(PulseAlpha(220), borderColor), 2))
            args.Graphics.DrawRoundRectangle(pen, panel, 10);
        DrawText(args, "POSITION RECOVERY MAP — VISUAL ONLY", panel.X + 11, panel.Y + 8, borderColor);

        if (!r.Active)
        {
            DrawStatsText(args, r.Status, panel.X + 11, panel.Y + 34);
            return;
        }

        string side = r.Side == Side.Buy ? "LONG" : "SHORT";
        string sizeState = r.MaxSize ? "  MAX SIZE" : r.SizeWarning ? "  SIZE WARNING" : string.Empty;
        DrawStatsText(args,
            $"{r.SymbolName}  {side}  {r.Quantity:N0}/{Math.Max(1, MaximumPositionContracts)} contracts  {r.DollarsPerPoint:N2}/point{sizeState}",
            panel.X + 11, panel.Y + 34);
        DrawStatsText(args,
            $"Entry/Fib 0 {FormatChartPrice(r.AverageEntryPrice)}  |  Worst/Fib 100 {FormatChartPrice(r.WorstPrice)}",
            panel.X + 11, panel.Y + 53);
        DrawStatsText(args,
            $"Campaign {SignedMoney(r.CampaignNetPnl, s.Currency)} | Closed {SignedMoney(r.CampaignRealizedPnl, s.Currency)} | Open {SignedMoney(r.NetPnl, s.Currency)}",
            panel.X + 11, panel.Y + 72);
        DrawStatsText(args,
            $"MFE {SignedMoney(r.BestNetPnl, s.Currency)} | MAE -{Money(Math.Abs(r.WorstNetPnl), s.Currency)} | Net B/E {FormatChartPrice(r.BreakEvenPrice)}",
            panel.X + 11, panel.Y + 91);
        DrawStatsText(args,
            $"Hard line {FormatChartPrice(r.HardLossPrice)} | {r.PointsToLossCap:N2} points to -{Money(PositionLossCap, s.Currency)}",
            panel.X + 11, panel.Y + 110);

        if (ProjectedAddContracts > 0)
        {
            Color previewColor = r.ProjectedExceedsMax ? DangerColor : r.ProjectedQuantity >= MaximumPositionContracts ? WarningColor : Color.Plum;
            using var brush = new SolidBrush(previewColor);
            string preview = r.ProjectedExceedsMax
                ? $"ADD +{ProjectedAddContracts}: {r.ProjectedQuantity:N0} contracts — EXCEEDS MAX {MaximumPositionContracts}"
                : $"ADD +{ProjectedAddContracts}: {r.ProjectedQuantity:N0} @ avg {FormatChartPrice(r.ProjectedAveragePrice)} | {r.ProjectedPointsToLossCap:N2} pts to cap";
            args.Graphics.DrawString(preview, StatsFont, brush, panel.X + 11, panel.Y + 129);
        }

        Color guidanceColor = r.LossDanger ? DangerColor : r.LossWarning ? WarningColor : Color.LightGray;
        DrawText(args, r.WarningText, panel.X + 11, panel.Y + 148, guidanceColor);
    }

    private void DrawMainPanel(PaintChartEventArgs args, DashboardSnapshot s)
    {
        Rectangle chart = args.Rectangle;
        float width = Math.Min(390, Math.Max(120, chart.Width - 32));
        var panel = new RectangleF(chart.X + 16, chart.Y + 16, width, 266);
        using (var brush = new LinearGradientBrush(panel, PanelBackground, Color.FromArgb(130, PanelBackground), LinearGradientMode.Vertical))
            args.Graphics.FillRoundRectangle(brush, panel, 12);
        using (var pen = new Pen(Color.FromArgb(PulseAlpha(210), StatusColor(s)), 2))
            args.Graphics.DrawRoundRectangle(pen, panel, 12);
        using (var brush = new SolidBrush(Color.White))
            args.Graphics.DrawString("SESSION ANGEL", HeaderFont, brush, panel.X + 14, panel.Y + 9);

        float y = panel.Y + 40;
        const float line = 19;
        DrawText(args, $"Account: {s.AccountName}", panel.X + 14, y, s.HistoryReady ? Color.White : WarningColor); y += line;
        DrawText(args, $"Data: {s.DataStatus}", panel.X + 14, y, s.HistoryReady ? SafeColor : WarningColor); y += line;
        DrawText(args, $"Active session: {s.ActiveSession}", panel.X + 14, y, Color.LightBlue); y += line;
        DrawText(args, $"Session P&L: {Money(s.SessionPnl, s.Currency)}", panel.X + 14, y, s.SessionPnl >= 0 ? SafeColor : DangerColor); y += line;
        DrawText(args, $"Realized: {Money(s.RealizedPnl, s.Currency)}", panel.X + 14, y, s.RealizedPnl >= 0 ? SafeColor : DangerColor); y += line;
        DrawText(args, $"Unrealized: {Money(s.UnrealizedPnl, s.Currency)}", panel.X + 14, y, s.UnrealizedPnl >= 0 ? SafeColor : DangerColor); y += line;
        DrawText(args, $"Session MFE: {SignedMoney(s.SessionMfe, s.Currency)} | MAE: -{Money(s.SessionMae, s.Currency)}", panel.X + 14, y, Color.LightCyan); y += line;
        DrawText(args, $"Peak-to-trough DD: {Money(s.MaxDrawdown, s.Currency)}", panel.X + 14, y, s.MaxDrawdown > 0 ? WarningColor : SafeColor); y += line;
        DrawText(args, $"Previous realized: {Money(s.PreviousRealizedPnl, s.Currency)}", panel.X + 14, y, s.PreviousRealizedPnl >= 0 ? SafeColor : DangerColor); y += line + 2;
        DrawText(args, $"Warning {Money(WarningThreshold, s.Currency)} | Limit {Money(DailyLossLimit, s.Currency)}", panel.X + 14, y, WarningColor);
    }

    private void DrawStatusIndicator(PaintChartEventArgs args, DashboardSnapshot s)
    {
        Rectangle rect = args.Rectangle;
        var circle = new RectangleF(Math.Max(rect.X + 4, rect.Right - 98), rect.Y + 20, 68, 68);
        Color color = StatusColor(s);
        string text = !s.HistoryReady ? "NO DATA"
            : s.Danger ? "DANGER"
            : s.Recovery.LossDanger ? "POS LOSS"
            : s.Recovery.MaxSize ? "MAX SIZE"
            : s.Warning || s.Recovery.LossWarning || s.Recovery.SizeWarning ? "WARNING"
            : "SAFE";
        int alpha = s.Danger || s.Warning || s.Recovery.HasWarning ? PulseAlpha(220) : 220;
        using (var brush = new SolidBrush(Color.FromArgb(Math.Min(alpha, 170), color))) args.Graphics.FillEllipse(brush, circle);
        using (var pen = new Pen(Color.FromArgb(alpha, color), 3)) args.Graphics.DrawEllipse(pen, circle);
        using var format = new StringFormat { Alignment = StringAlignment.Center };
        using var textBrush = new SolidBrush(color);
        args.Graphics.DrawString(text, BodyFont, textBrush, new RectangleF(circle.X - 12, circle.Bottom + 5, circle.Width + 24, 22), format);
    }

    private void DrawStatistics(PaintChartEventArgs args, DashboardSnapshot s, float y)
    {
        var panel = new RectangleF(args.Rectangle.X + 16, y, 300, 130);
        using (var bg = new SolidBrush(Color.FromArgb(180, 20, 20, 30))) args.Graphics.FillRoundRectangle(bg, panel, 10);
        DrawText(args, "CLOSED EXECUTIONS", panel.X + 10, panel.Y + 9, Color.LightBlue);
        float row = panel.Y + 34;
        DrawStatsText(args, $"Count: {s.ClosedExecutions}", panel.X + 10, row); row += 17;
        DrawStatsText(args, $"Wins / losses: {s.Wins} / {s.Losses}", panel.X + 10, row); row += 17;
        double winRate = s.ClosedExecutions == 0 ? 0 : s.Wins * 100.0 / s.ClosedExecutions;
        DrawStatsText(args, $"Win rate: {winRate:F1}%", panel.X + 10, row); row += 17;
        DrawStatsText(args, $"Largest win: {Money(s.LargestWin, s.Currency)}", panel.X + 10, row); row += 17;
        DrawStatsText(args, $"Largest loss: {Money(s.LargestLoss, s.Currency)}", panel.X + 10, row);
    }

    private void DrawPerformance(PaintChartEventArgs args, DashboardSnapshot s, float y)
    {
        var panel = new RectangleF(args.Rectangle.X + 330, y, 260, 130);
        using (var bg = new SolidBrush(Color.FromArgb(180, 30, 20, 20))) args.Graphics.FillRoundRectangle(bg, panel, 10);
        DrawText(args, "PERFORMANCE", panel.X + 10, panel.Y + 9, Color.LightCoral);
        DrawPerformanceBar(args, "P&L", s.SessionPnl, Math.Max(1, DailyLossLimit), panel.X + 10, panel.Y + 40);
        DrawPerformanceBar(args, "DD", -s.MaxDrawdown, Math.Max(1, MaxDrawdownLimit), panel.X + 10, panel.Y + 68);
        DrawPerformanceBar(args, "MFE", s.SessionMfe, Math.Max(1, DailyLossLimit), panel.X + 10, panel.Y + 96);
    }

    private void DrawPerformanceBar(PaintChartEventArgs args, string label, double value, double scale, float x, float y)
    {
        var background = new RectangleF(x + 48, y, 177, 15);
        using (var brush = new SolidBrush(Color.FromArgb(100, Color.Gray))) args.Graphics.FillRectangle(brush, background);
        float valueWidth = (float)(Math.Min(Math.Abs(value) / scale, 1) * background.Width);
        using (var brush = new SolidBrush(Color.FromArgb(190, value >= 0 ? SafeColor : WarningColor)))
            args.Graphics.FillRectangle(brush, new RectangleF(background.X, background.Y, valueWidth, background.Height));
        DrawStatsText(args, label, x, y);
    }

    private void DrawWarningOverlay(PaintChartEventArgs args, DashboardSnapshot s)
    {
        if (!s.Danger && !s.Warning && !s.Recovery.HasWarning) return;
        Rectangle rect = args.Rectangle;
        if (s.Danger || s.Recovery.LossDanger)
        {
            using (var overlay = new SolidBrush(Color.FromArgb(PulseAlpha(45), DangerColor))) args.Graphics.FillRectangle(overlay, rect);
            string message = s.Danger
                ? s.AccountLockRequested ? "DAILY LOSS LIMIT REACHED - ACCOUNT LOCK REQUESTED" : "DAILY LOSS LIMIT REACHED"
                : "POSITION LOSS CAP REACHED - VISUAL WARNING ONLY";
            SizeF size = args.Graphics.MeasureString(message, WarningFont);
            using var brush = new SolidBrush(Color.White);
            args.Graphics.DrawString(message, WarningFont, brush, rect.X + (rect.Width - size.Width) / 2, rect.Y + rect.Height / 2 - 25);
        }
        else
        {
            using var pen = new Pen(Color.FromArgb(PulseAlpha(130), WarningColor), 4);
            args.Graphics.DrawRectangle(pen, rect.X + 2, rect.Y + 2, Math.Max(0, rect.Width - 4), Math.Max(0, rect.Height - 4));
        }
    }

    private Color StatusColor(DashboardSnapshot s)
        => !s.HistoryReady ? WarningColor
            : s.Danger || s.Recovery.LossDanger || s.Recovery.MaxSize ? DangerColor
            : s.Warning || s.Recovery.LossWarning || s.Recovery.SizeWarning ? WarningColor
            : SafeColor;

    private int PulseAlpha(int maximum)
    {
        double pulse = 0.65 + 0.35 * ((Math.Sin(DateTime.UtcNow.TimeOfDay.TotalSeconds * Math.Max(1, AnimationSpeed) * 2) + 1) / 2);
        return Math.Clamp((int)(maximum * pulse), 0, 255);
    }

    private void DrawText(PaintChartEventArgs args, string text, float x, float y, Color color)
    {
        using var brush = new SolidBrush(color);
        args.Graphics.DrawString(text, BodyFont, brush, x, y);
    }

    private void DrawStatsText(PaintChartEventArgs args, string text, float x, float y)
    {
        using var brush = new SolidBrush(Color.LightGray);
        args.Graphics.DrawString(text, StatsFont, brush, x, y);
    }

    private Font HeaderFont => headerFont ?? SystemFonts.DefaultFont;
    private Font BodyFont => bodyFont ?? SystemFonts.DefaultFont;
    private Font WarningFont => warningFont ?? SystemFonts.DefaultFont;
    private Font StatsFont => statsFont ?? SystemFonts.DefaultFont;
    private static string Money(double value, string currency)
        => string.IsNullOrWhiteSpace(currency) ? $"{value:N2}" : $"{value:N2} {currency}";

    private static string SignedMoney(double value, string currency)
    {
        string sign = value > 0 ? "+" : string.Empty;
        return sign + Money(value, currency);
    }

    private string FormatChartPrice(double price)
    {
        if (!double.IsFinite(price))
            return "n/a";
        try
        {
            return Symbol?.FormatPrice(price) ?? price.ToString($"F{Math.Max(0, Digits)}");
        }
        catch
        {
            return price.ToString("N2");
        }
    }

    protected override void OnClear()
    {
        disposed = true;
        refreshTimer?.Dispose();
        refreshTimer = null;
        Core.Instance.TradeAdded -= OnTradeAdded;
        Core.Instance.PositionAdded -= OnPositionAdded;
        Core.Instance.PositionRemoved -= OnPositionRemoved;
        Position[] subscriptions;
        lock (stateLock)
        {
            subscriptions = observedPositions.ToArray();
            observedPositions.Clear();
        }
        foreach (Position position in subscriptions)
            position.Updated -= OnPositionUpdated;
        KeyValuePair<Trade, Action>[] tradeSubscriptions;
        lock (stateLock)
        {
            tradeSubscriptions = observedTrades.ToArray();
            observedTrades.Clear();
        }
        foreach (KeyValuePair<Trade, Action> subscription in tradeSubscriptions)
            subscription.Key.Updated -= subscription.Value;
        DisposeFonts();
        base.OnClear();
    }

    private void DisposeFonts()
    {
        headerFont?.Dispose(); bodyFont?.Dispose(); warningFont?.Dispose(); statsFont?.Dispose();
        headerFont = bodyFont = warningFont = statsFont = null;
    }

    public double CurrentSessionPnL { get { lock (stateLock) return currentSessionPnl; } }
    public double MaxDrawdown { get { lock (stateLock) return risk.MaxDrawdown; } }
    public double MaxProfit { get { lock (stateLock) return risk.SessionPeak; } }
    public bool IsWarningActive { get { lock (stateLock) return warningActive; } }
    public bool IsDangerActive { get { lock (stateLock) return dangerActive; } }
    public string CurrentActiveSession { get { lock (stateLock) return currentActiveSession; } }
    public double MFE { get { lock (stateLock) return risk.SessionPeak; } }
    public new double MAE { get { lock (stateLock) return risk.MaximumAdverseExcursion; } }

    private enum AlertKind
    {
        None,
        Warning,
        Drawdown,
        Danger,
        PositionWarning,
        MaxSize,
        PositionLoss
    }

    private readonly record struct TradeLedgerEntry(
        DateTime TimeUtc,
        double Pnl,
        bool HasPnl,
        bool IsClose,
        string PositionId,
        Symbol Instrument);

    private readonly record struct PositionAggregate(
        string LifecycleKey,
        string SymbolName,
        Side Side,
        double Quantity,
        double AverageEntryPrice,
        double CurrentPrice,
        double NetPnl,
        double TickSize,
        double TickCost,
        bool IsHedged,
        Symbol Instrument,
        string[] PositionIds,
        DateTime OpenTimeUtc)
    {
        public bool IsUsable => !IsHedged
            && Quantity > 0
            && double.IsFinite(AverageEntryPrice) && AverageEntryPrice > 0
            && double.IsFinite(CurrentPrice) && CurrentPrice > 0
            && double.IsFinite(NetPnl)
            && double.IsFinite(TickSize) && TickSize > 0
            && double.IsFinite(TickCost) && TickCost != 0;

        public static PositionAggregate Hedged(Symbol symbol, Account account)
            => new(
                string.Join("|", account.ConnectionId, account.Id, symbol.ConnectionId, symbol.Id, "hedged"),
                symbol.Name,
                Side.Buy,
                0,
                double.NaN,
                double.NaN,
                0,
                symbol.TickSize,
                0,
                true,
                symbol,
                Array.Empty<string>(),
                DateTime.MinValue);
    }

    private sealed class RecoveryMapState
    {
        public bool Visible { get; private set; }
        public bool Active { get; private set; }
        public string LifecycleKey { get; private set; } = string.Empty;
        public string SymbolName { get; private set; } = string.Empty;
        public string Status { get; private set; } = string.Empty;
        public Side Side { get; private set; }
        public double Quantity { get; private set; }
        public double AverageEntryPrice { get; private set; }
        public double WorstPrice { get; private set; }
        public double CurrentPrice { get; private set; }
        public double NetPnl { get; private set; }
        public double CampaignRealizedPnl { get; private set; }
        public double CampaignNetPnl { get; private set; }
        public double BestNetPnl { get; private set; }
        public double WorstNetPnl { get; private set; }
        public double TickSize { get; private set; }
        public double DollarsPerPoint { get; private set; }
        public double HardLossPrice { get; private set; }
        public double BreakEvenPrice { get; private set; }
        public double ProfitTargetPrice { get; private set; }
        public double Fib382Price { get; private set; }
        public double Fib500Price { get; private set; }
        public double Fib618Price { get; private set; }
        public double PointsToLossCap { get; private set; }
        public double ProjectedQuantity { get; private set; }
        public double ProjectedAveragePrice { get; private set; }
        public double ProjectedHardLossPrice { get; private set; }
        public double ProjectedPointsToLossCap { get; private set; }
        public bool ProjectedExceedsMax { get; private set; }

        public void Start(PositionAggregate position, double campaignRealizedPnl)
        {
            Visible = Active = true;
            LifecycleKey = position.LifecycleKey;
            SymbolName = position.SymbolName;
            Status = string.Empty;
            Side = position.Side;
            ApplyCurrent(position, campaignRealizedPnl);
            WorstPrice = Side == Side.Buy
                ? Math.Min(AverageEntryPrice, CurrentPrice)
                : Math.Max(AverageEntryPrice, CurrentPrice);
            BestNetPnl = Math.Max(0, CampaignNetPnl);
            WorstNetPnl = Math.Min(0, CampaignNetPnl);
        }

        public void Update(PositionAggregate position, double campaignRealizedPnl)
        {
            Visible = Active = true;
            ApplyCurrent(position, campaignRealizedPnl);
            WorstPrice = Side == Side.Buy
                ? Math.Min(WorstPrice, Math.Min(AverageEntryPrice, CurrentPrice))
                : Math.Max(WorstPrice, Math.Max(AverageEntryPrice, CurrentPrice));
            BestNetPnl = Math.Max(BestNetPnl, CampaignNetPnl);
            WorstNetPnl = Math.Min(WorstNetPnl, CampaignNetPnl);
        }

        private void ApplyCurrent(PositionAggregate position, double campaignRealizedPnl)
        {
            SymbolName = position.SymbolName;
            Quantity = position.Quantity;
            AverageEntryPrice = position.AverageEntryPrice;
            CurrentPrice = position.CurrentPrice;
            NetPnl = position.NetPnl;
            CampaignRealizedPnl = campaignRealizedPnl;
            CampaignNetPnl = CampaignRealizedPnl + NetPnl;
            TickSize = Math.Abs(position.TickSize);
            double tickCost = Math.Abs(position.TickCost);
            DollarsPerPoint = TickSize > 0 ? tickCost / TickSize * Quantity : 0;
        }

        public void CalculateLevels(double lossCap, double profitTarget, int projectedAdd, int maximumContracts)
        {
            double direction = Side == Side.Buy ? 1 : -1;
            if (!double.IsFinite(DollarsPerPoint) || DollarsPerPoint <= 0)
            {
                HardLossPrice = BreakEvenPrice = ProfitTargetPrice = double.NaN;
                return;
            }

            HardLossPrice = PriceForCampaignPnl(-lossCap, DollarsPerPoint, direction);
            BreakEvenPrice = PriceForCampaignPnl(0, DollarsPerPoint, direction);
            ProfitTargetPrice = PriceForCampaignPnl(profitTarget, DollarsPerPoint, direction);
            PointsToLossCap = Math.Max(0, (lossCap - Math.Max(0, -CampaignNetPnl)) / DollarsPerPoint);

            Fib382Price = AverageEntryPrice + (WorstPrice - AverageEntryPrice) * 0.382;
            Fib500Price = AverageEntryPrice + (WorstPrice - AverageEntryPrice) * 0.5;
            Fib618Price = AverageEntryPrice + (WorstPrice - AverageEntryPrice) * 0.618;

            ProjectedQuantity = Quantity + projectedAdd;
            ProjectedAveragePrice = ProjectedQuantity > 0
                ? (Quantity * AverageEntryPrice + projectedAdd * CurrentPrice) / ProjectedQuantity
                : AverageEntryPrice;
            double perContractPerPoint = DollarsPerPoint / Quantity;
            double projectedDollarsPerPoint = perContractPerPoint * ProjectedQuantity;
            ProjectedHardLossPrice = PriceForCampaignPnl(-lossCap, projectedDollarsPerPoint, direction);
            ProjectedPointsToLossCap = projectedDollarsPerPoint > 0
                ? Math.Max(0, (lossCap - Math.Max(0, -CampaignNetPnl)) / projectedDollarsPerPoint)
                : 0;
            ProjectedExceedsMax = ProjectedQuantity > maximumContracts;
        }

        private double PriceForCampaignPnl(double targetPnl, double dollarsPerPoint, double direction)
            => CurrentPrice + direction * (targetPnl - CampaignNetPnl) / dollarsPerPoint;

        public void SetUnavailable(string symbolName, string status)
        {
            Reset();
            Visible = true;
            SymbolName = symbolName;
            Status = status;
        }

        public void Reset()
        {
            Visible = Active = false;
            LifecycleKey = SymbolName = Status = string.Empty;
            Quantity = AverageEntryPrice = WorstPrice = CurrentPrice = 0;
            NetPnl = CampaignRealizedPnl = CampaignNetPnl = 0;
            BestNetPnl = WorstNetPnl = TickSize = DollarsPerPoint = 0;
            HardLossPrice = BreakEvenPrice = ProfitTargetPrice = double.NaN;
            Fib382Price = Fib500Price = Fib618Price = double.NaN;
            PointsToLossCap = ProjectedQuantity = ProjectedAveragePrice = 0;
            ProjectedHardLossPrice = double.NaN;
            ProjectedPointsToLossCap = 0;
            ProjectedExceedsMax = false;
        }

        public RecoverySnapshot Capture(bool sizeWarning, bool maxSize, bool lossWarning, bool lossDanger)
        {
            string warningText = lossDanger ? "Loss cap reached. No order was sent."
                : maxSize ? "Maximum position size reached. No order was sent."
                : lossWarning ? "Position loss is approaching the configured cap."
                : sizeWarning ? "Working-size warning: exposure has increased."
                : "Fibonacci levels are anchored to the worst price since entry.";
            return new RecoverySnapshot(
                Visible, Active, SymbolName, Status, Side, Quantity,
                AverageEntryPrice, WorstPrice, CurrentPrice,
                NetPnl, CampaignRealizedPnl, CampaignNetPnl,
                BestNetPnl, WorstNetPnl, TickSize, DollarsPerPoint,
                HardLossPrice, BreakEvenPrice, ProfitTargetPrice,
                Fib382Price, Fib500Price, Fib618Price, PointsToLossCap,
                ProjectedQuantity, ProjectedAveragePrice, ProjectedHardLossPrice,
                ProjectedPointsToLossCap, ProjectedExceedsMax,
                sizeWarning, maxSize, lossWarning, lossDanger, warningText);
        }
    }

    private sealed record RecoveryLine(
        string Label,
        double Price,
        double CampaignPnl,
        Color Color,
        DashStyle DashStyle,
        float Width);

    private readonly record struct RecoverySnapshot(
        bool Visible,
        bool Active,
        string SymbolName,
        string Status,
        Side Side,
        double Quantity,
        double AverageEntryPrice,
        double WorstPrice,
        double CurrentPrice,
        double NetPnl,
        double CampaignRealizedPnl,
        double CampaignNetPnl,
        double BestNetPnl,
        double WorstNetPnl,
        double TickSize,
        double DollarsPerPoint,
        double HardLossPrice,
        double BreakEvenPrice,
        double ProfitTargetPrice,
        double Fib382Price,
        double Fib500Price,
        double Fib618Price,
        double PointsToLossCap,
        double ProjectedQuantity,
        double ProjectedAveragePrice,
        double ProjectedHardLossPrice,
        double ProjectedPointsToLossCap,
        bool ProjectedExceedsMax,
        bool SizeWarning,
        bool MaxSize,
        bool LossWarning,
        bool LossDanger,
        string WarningText)
    {
        public bool HasWarning => SizeWarning || MaxSize || LossWarning || LossDanger;

        public double CampaignPnlAt(double price)
        {
            if (!Active || DollarsPerPoint <= 0 || !double.IsFinite(price))
                return double.NaN;
            double direction = Side == Side.Buy ? 1 : -1;
            return CampaignNetPnl + direction * (price - CurrentPrice) * DollarsPerPoint;
        }
    }

    private readonly record struct DashboardSnapshot(
        string AccountName, string Currency, string DataStatus, string ActiveSession,
        double RealizedPnl, double UnrealizedPnl, double SessionPnl, double SessionMfe,
        double SessionMae,
        double MaxDrawdown, double PreviousRealizedPnl, int ClosedExecutions, int Wins,
        int Losses, double LargestWin, double LargestLoss, bool HistoryReady,
        bool Warning, bool Danger, bool AccountLockRequested, RecoverySnapshot Recovery);
}

internal sealed class RiskAccumulator
{
    public double SessionPeak { get; private set; }
    public double SessionLow { get; private set; }
    public double MaxDrawdown { get; private set; }
    public double MaximumAdverseExcursion => Math.Max(0, -SessionLow);

    public void Reset() => SessionPeak = SessionLow = MaxDrawdown = 0;

    public void Update(double sessionPnl)
    {
        if (!double.IsFinite(sessionPnl)) return;
        SessionPeak = Math.Max(SessionPeak, sessionPnl);
        SessionLow = Math.Min(SessionLow, sessionPnl);
        MaxDrawdown = Math.Max(MaxDrawdown, SessionPeak - sessionPnl);
    }
}

internal static class SessionAngelGraphicsExtensions
{
    public static void FillRoundRectangle(this Graphics graphics, Brush brush, RectangleF rect, float radius)
    {
        using GraphicsPath path = GetRoundRectPath(rect, radius);
        graphics.FillPath(brush, path);
    }

    public static void DrawRoundRectangle(this Graphics graphics, Pen pen, RectangleF rect, float radius)
    {
        using GraphicsPath path = GetRoundRectPath(rect, radius);
        graphics.DrawPath(pen, path);
    }

    private static GraphicsPath GetRoundRectPath(RectangleF rect, float radius)
    {
        var path = new GraphicsPath();
        radius = Math.Max(0.1f, Math.Min(radius, Math.Min(rect.Width, rect.Height) / 2));
        float diameter = radius * 2;
        path.AddArc(rect.X, rect.Y, diameter, diameter, 180, 90);
        path.AddArc(rect.Right - diameter, rect.Y, diameter, diameter, 270, 90);
        path.AddArc(rect.Right - diameter, rect.Bottom - diameter, diameter, diameter, 0, 90);
        path.AddArc(rect.X, rect.Bottom - diameter, diameter, diameter, 90, 90);
        path.CloseFigure();
        return path;
    }
}
