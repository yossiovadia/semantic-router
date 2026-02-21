# NB Trading Procedures — Internal Operations Manual

## Order Execution Policy

### Pre-Trade Checks
1. Verify client tier authorization (intern/finance/principal)
2. Check portfolio allocation limits against proposed trade
3. Run compliance screening (sanctions list, restricted securities)
4. Validate settlement capacity (T+1 for equities, T+2 for bonds)

### Execution Venues
- **Primary**: NYSE, NASDAQ (US equities)
- **Fixed Income**: Bloomberg, Tradeweb, MarketAxess
- **FX**: EBS, Reuters Matching
- **Derivatives**: CME, ICE

### Best Execution Requirements
- Must obtain at least 3 quotes for OTC transactions > $1M
- Algorithmic execution required for orders > 10,000 shares
- Dark pool usage limited to 20% of average daily volume
- All executions must be timestamped to microsecond precision

## Risk Management Procedures

### Daily Risk Report
Generated at 6:00 AM ET, distributed to:
- Portfolio managers (full report)
- Risk committee (summary dashboard)
- Compliance team (exception report)

### Position Limits
| Asset Class | Single Position Max | Sector Max |
|-------------|-------------------|------------|
| US Equities | 5% of AUM | 25% of AUM |
| International Equities | 3% of AUM | 15% of AUM |
| Corporate Bonds | 2% of AUM per issuer | 20% of AUM |
| Sovereign Bonds | 10% of AUM per country | N/A |

### Escalation Procedures
- Loss > 1% daily: Notify portfolio manager
- Loss > 2% daily: Notify risk committee
- Loss > 5% daily: Emergency risk committee meeting
- Breach of position limits: Immediate notification to compliance

## Technology Stack
- Order Management: Bloomberg AIM
- Risk Analytics: MSCI RiskMetrics
- Compliance Monitoring: Actimize
- Model Routing: vSR Semantic Gateway (Egress Inference Router)
