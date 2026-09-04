# Session Angel risk model

## Monetary inputs

- Realized P&L is the sum of broker-provided `Trade.NetPnl` values for the selected
  account during the configured risk day.
- Unrealized P&L is the sum of broker-provided `Position.NetPnL` values for all open
  positions in the selected account.
- Session P&L is realized plus unrealized P&L, in the account currency reported by
  Quantower.
- Closed-execution statistics include closing fills (`PositionImpactType.Close`).
  A partial close is therefore one closed execution, not one complete position.

The completeness and accounting convention of these values ultimately depend on
the connected broker and its Quantower integration. If trade history cannot be
loaded, the indicator displays `NO DATA` / `Partial data` and never displays `SAFE`.

## Drawdown

For each observed account-P&L snapshot:

```text
sessionPeak = max(previousSessionPeak, sessionPnl)
currentDrawdown = sessionPeak - sessionPnl
maximumDrawdown = max(previousMaximumDrawdown, currentDrawdown)
```

This is peak-to-trough session drawdown. Account state is polled once per second and
also refreshed by chart, trade, position-open, and position-close events.

## Risk-day boundary

The default risk day starts at 09:30 New York time. The time-zone conversion uses
Windows/IANA time-zone data and therefore accounts for daylight-saving changes.
Risk alerts and maximum drawdown latch until the next configured risk day.

## Account locking

`Lock account at daily loss` is deliberately disabled by default. When enabled,
the first daily-loss breach calls Quantower's `Core.LockAccount` once. Session Angel
never unlocks an account automatically; unlocking requires an explicit manual risk
decision in Quantower. The indicator does not automatically flatten positions.

## Position recovery map

The recovery map is scoped to the selected account and the symbol of the chart on
which Session Angel is running. It is visualization and alerting only: it never
adds, closes, flattens, or modifies orders.

- Fib 0 always follows the broker's current weighted entry (`Position.OpenPrice`),
  so it moves whenever adding or reducing changes that entry.
- For a long position, Fib 100 is the lowest active-position price seen; for a
  short position, it is the highest. The 38.2%, 50%, and 61.8% levels are
  interpolated from the moving entry toward that worst-price anchor.
- Every line is labeled with estimated net campaign P&L. A campaign combines the
  broker `NetPnl` from closing executions belonging to the current position with
  the open position's broker `NetPnL`; both values preserve the broker's own
  futures lot-accounting method and transaction-cost treatment.
- The `NET B/E AFTER COSTS` line solves
  `closed-execution NetPnl + projected open-position NetPnL = 0`. Maximum-loss and
  profit-target prices use the same campaign equation rather than the gross
  weighted-entry price.
- Position MFE is the highest broker `NetPnL` observed during the open-position
  lifecycle. Position MAE is the absolute value of the lowest broker `NetPnL`
  observed. These dollar values change only when a new favorable/adverse extreme
  is reached. Session MFE and MAE use the same high-water/low-water behavior for
  total session P&L.
- The projected-add preview assumes the configured quantity fills at the current
  broker exit price. It shows the resulting weighted average and remaining price
  distance to the position loss cap, but does not send an order.

Only normal, non-zero broker positions are considered active. Fake or zero-quantity
position objects cannot keep recovery lines alive after a position has closed.

Closed executions are keyed by broker trade identity. A closing fill is counted as
soon as its `PositionImpactType` is `Close`; if its broker `NetPnl` arrives later on
`Trade.Updated`, the same ledger entry is revised rather than duplicated, and the
win/loss totals are recalculated from the revised value.

Session Angel refreshes the recovery map from trade events, position add/remove
events, every live `Position.Updated` event, chart updates, and the one-second safety
poll. Each event-driven refresh explicitly asks Quantower to redraw the chart.
