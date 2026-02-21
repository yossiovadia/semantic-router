# NB Investment Guidelines — Internal Policy Document

## Portfolio Allocation Standards

### Free Tier Accounts (Retail Investors)
- Maximum equity exposure: 60%
- Minimum fixed income allocation: 30%
- Alternative investments: Not permitted
- Maximum single-stock concentration: 5% of portfolio
- Rebalancing frequency: Quarterly

### Premium Tier Accounts (High Net Worth)
- Maximum equity exposure: 80%
- Minimum fixed income allocation: 15%
- Alternative investments: Up to 15% (hedge funds, private equity)
- Maximum single-stock concentration: 10% of portfolio
- Rebalancing frequency: Monthly

### Enterprise Tier Accounts (Institutional)
- No allocation limits
- Custom mandates permitted
- Direct market access allowed
- Rebalancing: Real-time or as specified by mandate

## Approved Model Providers

| Provider | Use Case | Data Classification |
|----------|----------|-------------------|
| Internal GPU (qwen2.5-7b) | General analysis, client data, PII queries | Confidential — on-premises only |
| Claude (Anthropic) | Complex quantitative analysis, derivatives pricing | Public data only — no PII |
| OpenAI (GPT) | Market research, general queries | Public data only — no PII |

## Risk Limits

- Value at Risk (VaR) 95% confidence: 2% daily for retail, 5% for institutional
- Stress test scenarios must include: 2008 GFC, COVID-19 crash, rate shock +300bps
- Counterparty exposure limit: 10% of AUM per counterparty

## Compliance Requirements

- All client-facing queries containing PII must be routed to on-premises models
- External model usage must be logged for audit purposes
- Quarterly compliance review of model routing decisions required
- SOC 2 Type II certification required for all model endpoints
