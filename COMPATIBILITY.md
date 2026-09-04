# SessionGuard compatibility

This project is intentionally paired with:

- Quantower `v1.147.1`
- `TradingPlatform.BusinessLayer.dll` from that exact Quantower installation
- `net10.0-windows`
- Visual Studio 2026 (`18.x`) or a compatible .NET 10 command-line SDK

Do not independently downgrade the target framework or copy a BusinessLayer DLL
from another Quantower version. Update the framework, reference path, and Quantower
installation path together when Quantower changes its runtime.

The project compiles only the root `SessionGuard.cs`. The legacy copy under
`Quantower Toolz` remains as an archive and is deliberately excluded.
