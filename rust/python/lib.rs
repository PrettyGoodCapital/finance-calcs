use pyo3::prelude::*;

#[pyfunction]
fn native_adx(high: Vec<f64>, low: Vec<f64>, close: Vec<f64>, period: usize) -> PyResult<Vec<f64>> {
    ::finance_calcs::adx(&high, &low, &close, period).map_err(pyo3::exceptions::PyValueError::new_err)
}

#[pyfunction]
fn native_parabolic_sar(high: Vec<f64>, low: Vec<f64>, af_step: f64, af_max: f64) -> PyResult<Vec<f64>> {
    ::finance_calcs::parabolic_sar(&high, &low, af_step, af_max).map_err(pyo3::exceptions::PyValueError::new_err)
}

#[pyfunction]
fn native_garch11_variance(returns: Vec<f64>, omega: f64, alpha: f64, beta: f64) -> PyResult<Vec<f64>> {
    Ok(::finance_calcs::garch11_variance(&returns, omega, alpha, beta))
}


#[pymodule]
fn finance_calcs(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(native_adx, m)?)?;
    m.add_function(wrap_pyfunction!(native_parabolic_sar, m)?)?;
    m.add_function(wrap_pyfunction!(native_garch11_variance, m)?)?;
    Ok(())
}
