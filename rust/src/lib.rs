pub fn garch11_variance(returns: &[f64], omega: f64, alpha: f64, beta: f64) -> Vec<f64> {
    if returns.is_empty() {
        return Vec::new();
    }
    let mut out = vec![0.0; returns.len()];
    let mut prev = returns.iter().map(|v| v * v).sum::<f64>() / returns.len() as f64;
    if !prev.is_finite() || prev <= 0.0 {
        prev = 1e-8;
    }
    for (i, r) in returns.iter().enumerate() {
        let value = omega + alpha * r * r + beta * prev;
        out[i] = value.max(0.0);
        prev = out[i];
    }
    out
}

pub fn parabolic_sar(
    high: &[f64],
    low: &[f64],
    af_step: f64,
    af_max: f64,
) -> Result<Vec<f64>, String> {
    if high.len() != low.len() {
        return Err("high and low lengths must match".to_string());
    }
    if high.is_empty() {
        return Ok(Vec::new());
    }
    let mut sar = vec![0.0; high.len()];
    let mut long = true;
    let mut af = af_step.max(1e-6);
    let mut ep = high[0];
    sar[0] = low[0];

    for i in 1..high.len() {
        let mut next = sar[i - 1] + af * (ep - sar[i - 1]);
        if long {
            next = next.min(low[i - 1]);
            if low[i] < next {
                long = false;
                next = ep;
                ep = low[i];
                af = af_step.max(1e-6);
            } else if high[i] > ep {
                ep = high[i];
                af = (af + af_step).min(af_max);
            }
        } else {
            next = next.max(high[i - 1]);
            if high[i] > next {
                long = true;
                next = ep;
                ep = high[i];
                af = af_step.max(1e-6);
            } else if low[i] < ep {
                ep = low[i];
                af = (af + af_step).min(af_max);
            }
        }
        sar[i] = next;
    }
    Ok(sar)
}

pub fn adx(high: &[f64], low: &[f64], close: &[f64], period: usize) -> Result<Vec<f64>, String> {
    if high.len() != low.len() || high.len() != close.len() {
        return Err("high, low, and close lengths must match".to_string());
    }
    if period == 0 {
        return Err("period must be > 0".to_string());
    }
    let n = high.len();
    if n == 0 {
        return Ok(Vec::new());
    }

    let mut tr = vec![0.0; n];
    let mut plus_dm = vec![0.0; n];
    let mut minus_dm = vec![0.0; n];
    for i in 1..n {
        let up = high[i] - high[i - 1];
        let down = low[i - 1] - low[i];
        plus_dm[i] = if up > down && up > 0.0 { up } else { 0.0 };
        minus_dm[i] = if down > up && down > 0.0 { down } else { 0.0 };
        let h_l = high[i] - low[i];
        let h_pc = (high[i] - close[i - 1]).abs();
        let l_pc = (low[i] - close[i - 1]).abs();
        tr[i] = h_l.max(h_pc).max(l_pc);
    }

    let alpha = 1.0 / period as f64;
    let mut tr_sm = vec![0.0; n];
    let mut plus_sm = vec![0.0; n];
    let mut minus_sm = vec![0.0; n];
    for i in 1..n {
        tr_sm[i] = alpha * tr[i] + (1.0 - alpha) * tr_sm[i - 1];
        plus_sm[i] = alpha * plus_dm[i] + (1.0 - alpha) * plus_sm[i - 1];
        minus_sm[i] = alpha * minus_dm[i] + (1.0 - alpha) * minus_sm[i - 1];
    }

    let mut dx = vec![0.0; n];
    for i in 0..n {
        let p_di = if tr_sm[i] > 0.0 {
            100.0 * plus_sm[i] / tr_sm[i]
        } else {
            0.0
        };
        let m_di = if tr_sm[i] > 0.0 {
            100.0 * minus_sm[i] / tr_sm[i]
        } else {
            0.0
        };
        let denom = p_di + m_di;
        dx[i] = if denom > 0.0 {
            100.0 * (p_di - m_di).abs() / denom
        } else {
            0.0
        };
    }

    let mut out = vec![0.0; n];
    for i in 1..n {
        out[i] = alpha * dx[i] + (1.0 - alpha) * out[i - 1];
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_garch11_positive() {
        let out = garch11_variance(&[0.01, -0.02, 0.015, 0.005], 1e-6, 0.1, 0.85);
        assert_eq!(out.len(), 4);
        assert!(out.iter().all(|v| *v >= 0.0));
    }

    #[test]
    fn test_parabolic_sar_shape() {
        let high = [10.0, 10.4, 10.6, 10.3, 10.8];
        let low = [9.8, 10.1, 10.2, 10.0, 10.4];
        let out = parabolic_sar(&high, &low, 0.02, 0.2).unwrap();
        assert_eq!(out.len(), high.len());
    }

    #[test]
    fn test_adx_bounds() {
        let high = [10.0, 10.4, 10.6, 10.3, 10.8, 11.0, 10.9];
        let low = [9.8, 10.1, 10.2, 10.0, 10.4, 10.6, 10.5];
        let close = [9.9, 10.2, 10.5, 10.1, 10.7, 10.8, 10.6];
        let out = adx(&high, &low, &close, 3).unwrap();
        assert_eq!(out.len(), high.len());
        assert!(out.iter().all(|v| *v >= 0.0 && *v <= 100.0));
    }
}
